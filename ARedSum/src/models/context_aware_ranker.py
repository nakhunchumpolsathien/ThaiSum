import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.encoder import TransformerEncoderLayer
from models.decoder import TransformerDecoderLayer, TransformerDecoder
from others.logging import logger

class HitMapBilinearMatchModel(nn.Module):
    '''Currently just for a pair'''
    def __init__(self, seg_count, ngram_segs=[20,20,20], bilinear_out=10):
        super(HitMapBilinearMatchModel, self).__init__()
        #the output dimension can also be more than 1
        self._sim_seg_count = seg_count
        self._grams_seg_counts = ngram_segs
        self.bilinear = nn.Bilinear(2, 2+self._sim_seg_count+sum(self._grams_seg_counts), bilinear_out)
        self.linear = nn.Linear(bilinear_out, 1)
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.bias = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, sent_group_scores, sel_sent_emb, sel_sent_masks, \
            group_embs, candi_sent_masks, raw_sent_embs=None,
            sel_sent_hit_map=None):
        '''
        #can only handle the case when group_size is set to 1
        sent_group_scores: batch_size, max_sent_count
        sel_sent_emb: batch_size, sel_sent_count, emb_dim
        group_emb: batch_size, max_group_count, emb_dim
        candi_sent_masks: batch_size, max_group_count
        '''
        if sel_sent_masks is None or sel_sent_hit_map is None:# or sel_sent_masks.sum() == 0
            #this is for inference
            #(batch_size, max_sent_count, self._seg_count + 20+20+4)
            h_3 = sent_group_scores + self.bias# only forward, backpropagate cannot be done if no parameter is included (variables that require_grad = true)
        else:
            sent_group_scores = torch.softmax(sent_group_scores, -1) #dim=-1
            batch_size, max_sent_count, _ = group_embs.size()
            _, sel_sent_count, _ = sel_sent_emb.size()
            sent_score_seg = sent_group_scores.new_zeros(list(sent_group_scores.size()) + [2])
            #sent_group_scores: batch_size, max_sent_count, 2 #dim 0 for masked value, dim 1 for true value
            sent_seg_idx = torch.where(candi_sent_masks==0, torch.tensor(0).to(sent_group_scores.device),
                                        torch.tensor(1).to(sent_group_scores.device)).long()
            sent_score_seg[torch.arange(sent_group_scores.size()[0]).unsqueeze(-1),
                    torch.arange(sent_group_scores.size()[1]),
                    sent_seg_idx] = sent_group_scores

            sel_sent_emb = sel_sent_emb.unsqueeze(1).expand(-1, max_sent_count, -1, -1)
            group_embs = group_embs.unsqueeze(2).expand(-1, -1, sel_sent_count, -1)
            #(batch_size, max_sent_count, sel_sent_count, emb_dim)
            h_1 = self.cosine(group_embs.contiguous(), sel_sent_emb.contiguous())

            #(batch_size, max_sent_count, sel_sent_count)
            h_1 = torch.max(h_1, dim = -1)[0]
            #(batch_size, max_sent_count)
            h_1_new = torch.where(candi_sent_masks==1, h_1, torch.tensor(float('-inf')).to(h_1.device))
            h_1_max = torch.max(h_1_new, dim=1, keepdim=True)[0]
            h_1_new = torch.where(candi_sent_masks==1, h_1, torch.tensor(float('inf')).to(h_1.device))
            h_1_min = torch.min(h_1_new, dim=1, keepdim=True)[0]
            h_1_portion = (h_1 - h_1_min) / (h_1_max - h_1_min + 1e-9)
            #(batch_size, max_sent_count)
            h_1_seg = h_1.new_zeros(list(h_1.size()) + [self._sim_seg_count + 2]) #0 and 1 different from the others
            #try to differentiate the case cosine_sim equals to 0 or 1 with the case of 0,1 from min-max normalization (mininum and max value)
            h_1_seg_idx_raw = (h_1_portion / (1.0/self._sim_seg_count) + 1).long()
            h_1_seg_idx_raw = torch.where(h_1_seg_idx_raw==self._sim_seg_count + 1, torch.tensor(self._sim_seg_count).to(h_1.device), h_1_seg_idx_raw)
            h_1_seg_idx_0 = torch.where(h_1==0, torch.tensor(0).to(h_1.device), h_1_seg_idx_raw)
            h_1_seg_idx = torch.where(h_1==1, torch.tensor(self._sim_seg_count + 1).to(h_1.device), h_1_seg_idx_0)

            # max of h1_portion will be a little less than 1 because 1e-9 in the denominator, so idx will be at most self._seg_count - 1
            h_1_seg[torch.arange(h_1.size()[0]).unsqueeze(-1),
                    torch.arange(h_1.size()[1]),
                    h_1_seg_idx] = 1.
            #(batch_size, max_sent_count, self._seg_count+2)
            div_pattern = torch.cat([h_1_seg] + sel_sent_hit_map, dim=-1) #sel_sent_hit_map is [(batch_size, sent_count, seg_count) for 1,2,3 gram]
            #(batch_size, max_sent_count, self._seg_count + 20+20+4)
            h_2 = self.bilinear(sent_score_seg.contiguous(), div_pattern.contiguous())
            h_3 = self.linear(torch.tanh(h_2)).squeeze(-1)
            #(batch_size, max_sent_count, 1)
        #(batch_size, max_sent_count, group_size)
        sent_scores = h_3 * candi_sent_masks.float()
        return sent_scores


class biLinearModel(nn.Module):
    '''Currently just for a pair'''
    def __init__(self, hidden_size):
        super(biLinearModel, self).__init__()
        #the output dimension can also be more than 1
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, doc_emb, group_embs, candi_sent_masks):
        '''
        doc_emb: batch_size, 1, emb_dim
        group_emb: batch_size, max_sent_count, emb_dim
        candi_sent_masks: batch_size, max_group_count
        '''
        doc_emb = doc_emb.expand_as(group_embs)
        #(batch_size, max_sent_count, emb_dim)
        h_0 = self.bilinear(group_embs.contiguous(), doc_emb.contiguous())
        #(batch_size, max_sent_count, 1)

        sent_group_scores = h_0.squeeze(-1) * candi_sent_masks.float()
        return sent_group_scores

class PairwiseMLP(nn.Module):
    '''input vectors
        Doc: document embeddings (batch_size, emb_dim),
        sel_sents: Selected sentences (batch_size, n_sents, emb_dim)
        Sentence Group: (batch_size, group_size, emb_dim)
    '''
    def __init__(self, hidden_size, args):
        super(PairwiseMLP, self).__init__()
        self.selector = None
        self.model_name = args.model_name
        if self.model_name == "base":
            self.scorer = biLinearModel(hidden_size)
        elif self.model_name == "ctx":
            self.scorer = biLinearModel(hidden_size)
            self.scorer.eval()
            ngram_segs = [int(x) for x in args.ngram_seg_count.split(',')]
            self.selector = HitMapBilinearMatchModel(args.seg_count, ngram_segs, args.bilinear_out)

    def forward(self, doc_emb, sent_embs, sel_sent_idxs, sel_sent_masks, \
            group_idxs, candi_sent_masks, is_test=False, raw_sent_embs=None,
            sel_sent_hit_map=None):
        '''
        sent_embs: (batch_size, sent_count, emb_dim)
        sel_sent_idxs: (batch_size, sel_sent_count)
        doc_emb: (batch_size, 1, emb_dim)
        group_idxs: (batch_size, max_sent_count)
        '''
        _, _, emb_dim = doc_emb.size()
        batch_size, max_sent_count = group_idxs.size()

        #although it should be fine if we do not append zero vectors since mask is applied later
        if sel_sent_idxs is None or len(sel_sent_idxs) == 0 or len(sel_sent_idxs[0]) == 0: #no selected sentences
            sel_sent_emb = torch.zeros(batch_size, 1, emb_dim).to(sent_embs.device)
            sel_sent_masks = torch.zeros(batch_size, 1).to(sent_embs.device)
        else:
            sent_vec_for_sel = sent_embs
            #whether we need this? does it make difference?
            #if self.model_name == 'ctx':
                #sent_vec_for_sel = raw_sent_embs

            sel_sent_emb = torch.cat([sent_vec_for_sel[i][sel_sent_idxs[i]].unsqueeze(0)\
                    for i in range(batch_size)], dim=0)
            sel_sent_emb = sel_sent_emb * sel_sent_masks[:,:,None].float()
        group_embs = sent_embs[torch.arange(batch_size).unsqueeze(1), group_idxs]
        group_embs = group_embs * candi_sent_masks[:,:,None].float()
        sent_group_scores = self.scorer(doc_emb, group_embs, candi_sent_masks)
        if self.selector is not None:
            sent_group_scores = self.selector(sent_group_scores, sel_sent_emb, sel_sent_masks,
                    group_embs, candi_sent_masks, raw_sent_embs,
                    sel_sent_hit_map)

        #sent_scores = sent_group_scores * candi_masks.float()
        #batch_size, max_sent_count, group_size
        return sent_group_scores
