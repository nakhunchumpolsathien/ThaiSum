
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.encoder import TransformerInterEncoder, \
        TransformerInterEncoderClassifier, Classifier, RNNEncoder
from models.seq2seq import TransformerDecoderSeq
from models.context_aware_ranker import PairwiseMLP
from models.optimizers import Optimizer
import models.data_util as du
import numpy as np


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '' and checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '' and checkpoint is not None:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Bert(nn.Module):
    def __init__(self, temp_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()
        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained('bert-base-multilingual-cased', cache_dir=temp_dir)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask =mask)
        top_vec = encoded_layers[-1]
        return top_vec



class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, bert_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.temp_dir, load_pretrained_bert, bert_config)

        self.transformer_encoder = TransformerInterEncoder(self.bert.model.config.hidden_size, args.ff_size, args.heads, args.dropout, args.inter_layers)
        if (args.model_name == "seq"):
            self.encoder = TransformerDecoderSeq(
                    self.bert.model.config.hidden_size, args.ff_size, args.heads,
                    args.dropout, args.inter_layers, args.use_doc)
        else:
        #if ('ctx' in args.model_name or 'base' in args.model_name):
            self.encoder = PairwiseMLP(
                    self.bert.model.config.hidden_size, args)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
            for p in self.transformer_encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)

        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in self.transformer_encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)
    def load_cp(self, pt, strict=True):
        self.load_state_dict(pt['model'], strict=strict)

    def infer_sentences(self, batch, num_sent, stats=None):
        with torch.no_grad():
            src, labels, segs = batch.src, batch.labels, batch.segs
            clss, mask, mask_cls = batch.clss, batch.mask, batch.mask_cls
            #group_idxs, pair_masks = batch.test_groups, batch.test_pair_masks
            group_idxs = batch.groups
            #shouldn't use this hit_map and mask, these are for random selected indices
            #should compute new himap

            sel_sent_idxs = torch.LongTensor([[] for i in range(batch.batch_size)]).to(labels.device)
            sel_sent_masks = torch.LongTensor([[] for i in range(batch.batch_size)]).to(labels.device)
            candi_masks = mask_cls.clone().detach()
            top_vec = self.bert(src, segs, mask)
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
            raw_sents_vec = sents_vec
            doc_emb, sents_vec = self.transformer_encoder(raw_sents_vec, mask_cls)

            ngram_segs = [int(x) for x in self.args.ngram_seg_count.split(',')]
            for sent_id in range(num_sent):
                hit_map = None #initially be none
                if sent_id > 0 and self.args.model_name == 'ctx':
                    hit_map = du.get_hit_ngram(batch.src_str, sel_sent_idxs, sel_sent_masks, ngram_segs)
                sent_scores = self.encoder(doc_emb, sents_vec, sel_sent_idxs,
                        sel_sent_masks, group_idxs, candi_masks,
                        is_test=True, raw_sent_embs=raw_sents_vec,
                        sel_sent_hit_map=hit_map)

                sent_scores[candi_masks==False] = float('-inf')
                #in case illegal values exceed 1000
                sent_scores = sent_scores.cpu().data.numpy()
                #print(sent_scores)
                sorted_ids = np.argsort(-sent_scores, 1)
                #batch_size, sorted_sent_ids
                cur_selected_ids = torch.tensor(sorted_ids[:,0]).unsqueeze(-1).to(labels.device)
                cur_masks = torch.ones(batch.batch_size, 1).long().to(labels.device)

                sel_sent_idxs = torch.cat([sel_sent_idxs, cur_selected_ids], dim=1)
                sel_sent_masks = torch.cat([sel_sent_masks, cur_masks], dim=1)
                du.set_selected_sent_to_value(candi_masks, sel_sent_idxs, sel_sent_masks, False)

            return sel_sent_idxs, sel_sent_masks


    def forward(self, x, mask, segs, clss, mask_cls, group_idxs,
            sel_sent_idxs=None, sel_sent_masks=None, candi_sent_masks=None, is_test=False,
            sel_sent_hit_map=None):
        top_vec = self.bert(x, segs, mask)
        #top_vec is batch_size, sequence_length, embedding_size
        #get the embedding of each CLS symbol in the batch
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        raw_sents_vec = sents_vec
        doc_emb, sents_vec = self.transformer_encoder(raw_sents_vec, mask_cls)
        sent_scores = self.encoder(doc_emb, sents_vec, sel_sent_idxs,
                sel_sent_masks, group_idxs, candi_sent_masks, is_test,
                raw_sent_embs=raw_sents_vec,
                sel_sent_hit_map=sel_sent_hit_map)
        #batch_size, max_sent_count
        return sent_scores, mask_cls
