import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward
from models.decoder import TransformerDecoder, \
        TransformerDecoderLayer, get_subsequent_mask

class TransformerDecoderSeq(nn.Module):
    '''Only apply to when groupsize == 1'''
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers, use_doc=False, aggr='last'):
        super(TransformerDecoderSeq, self).__init__()
        self.aggr = aggr
        self.decoder = TransformerDecoder(d_model, d_ff, heads, dropout, num_inter_layers)
        self.dropout = nn.Dropout(dropout)
        self.use_doc = use_doc
        if self.use_doc:
            self.linear_doc1 = nn.Linear(2 * d_model, d_model)
            self.linear_doc2 = nn.Linear(d_model, 1)
            self.bilinear = nn.Bilinear(d_model, d_model, 1)
        self.linear_sent1 = nn.Linear(2 * d_model, d_model)
        self.linear_sent2 = nn.Linear(d_model, 1)
        self.linear = nn.Linear(2, 1)
        self.start_emb = torch.nn.Parameter(torch.rand(1, d_model))

    def translate(self, doc_emb, encoder_output_sents_vec, sel_sent_idxs, sel_sent_masks, \
            mask_cls, raw_sent_embs):
        '''
        doc_emb: batch_size, 1, emb_dim
        sel_sent_idxs: batch_size, sel_sent_count
        sel_sent_masks: batch_size, sel_sent_count
        encoder_output_sents_vec: batch_size, sent_count, emb_dim
                    a group of sentences for comparison and scoring
                    #encoder_output
        masks_cls: : batch_size, sent_count
                    #encoder_output_masks
        output      : a group of scores
                    batch_size, sent_count
        '''
        encoder_outputs = encoder_output_sents_vec * mask_cls[:,:,None].float()
        batch_size, sent_count, emb_dim = encoder_outputs.size()

        if self.use_doc:
            h_0 = self.bilinear(doc_emb.expand_as(encoder_outputs).contiguous(), encoder_outputs.contiguous())
            h_0 = torch.tanh(h_0)
            #doc_context = torch.cat((doc_emb.expand_as(encoder_outputs), encoder_outputs), dim=2)
            #(batch_size, sent_count, 2*emb_dim)
            #h_0 = self.linear_doc2(torch.sigmoid(self.linear_doc1(doc_context)))

        #first_step_embs = torch.zeros(batch_size, 1, emb_dim).to(encoder_outputs.device)
        first_step_embs = self.start_emb.unsqueeze(0).expand(batch_size, -1, -1).to(encoder_outputs.device)
        first_step_masks = torch.ones(batch_size, 1).to(encoder_outputs.device).byte()
        context_embs = first_step_embs
        context_masks = first_step_masks

        #print('sent_group_scores without selected', sent_group_scores)
        #print('linear_doc1', self.linear_doc1.weight.data)
        #print('linear_doc2', self.linear_doc2.weight.data)
        if sel_sent_idxs is not None and sel_sent_masks is not None \
            and 0 not in sel_sent_idxs.size() and 0 not in sel_sent_masks.size() \
            and sel_sent_masks.sum().item() > 0:
                #make sure there are selected sentences
            sel_sent_embs = raw_sent_embs[torch.arange(batch_size).unsqueeze(1), sel_sent_idxs]
            sel_sent_embs = sel_sent_embs * sel_sent_masks[:,:,None].float()
            context_embs = torch.cat((first_step_embs, sel_sent_embs), dim=1) #1+tgt_length
            context_masks = torch.cat((first_step_masks, sel_sent_masks.byte()), dim=1)

        dec_output = self.decoder(context_embs, context_masks, encoder_outputs, mask_cls)
        dec_output = dec_output[:,-1:,:]# the last embedding
        #batch_size, 1,emb_dim
        #h_1 = self.bilinear(dec_output.expand_as(raw_sent_embs).contiguous(), raw_sent_embs.contiguous())
        #batch_size, max_sent_count, 1
        sent_context = torch.cat((dec_output.expand_as(raw_sent_embs), raw_sent_embs), dim=2)
        h_1 = self.linear_sent2(torch.tanh(self.linear_sent1(sent_context)))
        #(batch_size, group_count, 1)
        sent_group_scores = h_1.squeeze(-1)
        if self.use_doc:
            sent_group_scores = self.linear(torch.cat((h_0, h_1), dim=-1)).squeeze(-1)
        #print('sent_group_scores selected', sent_group_scores)
        #(batch_size, group_count)
        sent_group_scores = sent_group_scores * mask_cls.float()
        #print('sent_group_scores', sent_group_scores.size())
        #print('sent_group_scores', sent_group_scores)
        return sent_group_scores
        #(batch_size, group_count)

    def decode(self, doc_emb, encoder_output_sents_vec, tgt_sent_idxs, \
            mask_cls, raw_sent_embs):
        '''
        queries: document embeddings or selected sentence embeddings
            doc_emb: batch_size, 1, emb_dim
            tgt_seq_sent_embs: batch_size, sel_sent_count, emb_dim
        group_embs  : batch_size, max_group_count, group_size, emb_dim
                    a group of sentences for comparison and scoring
                    #encoder_output
        mask_cls : batch_size, max_group_count
                    masks of the groups
        raw_sent_embs: batch_size, max_sent_count(max_group_count), emb_dim
        output      : a group of scores
                    batch_size, sent_count, group_size
        '''
        tgt_seq_masks = tgt_sent_idxs.gt(-1)
        encoder_outputs = encoder_output_sents_vec * mask_cls[:,:,None].float()
        batch_size, sent_count, emb_dim = encoder_outputs.size()
        tgt_seq_sent_embs = raw_sent_embs[torch.arange(batch_size).unsqueeze(1), tgt_sent_idxs]
        tgt_seq_sent_embs = tgt_seq_sent_embs * tgt_seq_masks[:,:,None].float()

        #first_step_embs = torch.zeros(batch_size, 1, emb_dim).to(tgt_seq_sent_embs.device)
        first_step_embs = self.start_emb.unsqueeze(0).expand(batch_size, -1, -1).to(encoder_outputs.device)
        first_step_masks = torch.ones(batch_size, 1).to(tgt_seq_sent_embs.device).byte()
        tgt_embs = torch.cat((first_step_embs, tgt_seq_sent_embs), dim=1) #1+tgt_length
        tgt_masks = torch.cat((first_step_masks, tgt_seq_masks), dim=1)
        sub_seq_masks = get_subsequent_mask(tgt_embs) # [[0,1,1],[0,0,1],[0,0,0]]

        if self.use_doc:
            h_0 = self.bilinear(doc_emb.expand_as(encoder_outputs).contiguous(), encoder_outputs.contiguous())
            h_0 = torch.tanh(h_0)
            #doc_context = torch.cat((doc_emb.expand_as(encoder_outputs), encoder_outputs), dim=2)
            #(batch_size, sent_count, 2*emb_dim)
            #h_0 = self.linear_doc2(torch.sigmoid(self.linear_doc1(doc_context)))

        dec_output = self.decoder(tgt_embs, tgt_masks, encoder_outputs, mask_cls, sub_seq_masks)
        # batch_size, tgt_seq_len + 1, emb_dim
        dec_output = dec_output[:, :-1, :] # 0, I, like -> I, like, it
        #otherwise, it should be 0,I,like,it -> I, like, it, <end>
        dec_output = dec_output.unsqueeze(2).expand(-1,-1,sent_count,-1)
        #h_1 = self.bilinear(dec_output.expand_as(raw_sent_embs).contiguous(), raw_sent_embs.contiguous())
        sent_context = torch.cat((dec_output, raw_sent_embs.unsqueeze(1).expand_as(dec_output)), dim=-1)
        # batch_size, tgt_seq_len, sent_count, 2*emb_dim
        h_1 = self.linear_sent2(torch.tanh(self.linear_sent1(sent_context)))
        #(batch_size, tgt_seq_len, sent_count, 1)
        seq_sent_scores = h_1.squeeze(-1)
        if self.use_doc:
            seq_sent_scores = self.linear(torch.cat((h_0.unsqueeze(1).expand_as(h_1), h_1), dim=-1)).squeeze(-1)
        # batch_size, tgt_seq_len, sent_count
        seq_sent_scores = seq_sent_scores * mask_cls[:,None,:].float()
        #print('seq_sent_scores', seq_sent_scores.size())
        #print('seq_sent_scores', seq_sent_scores)
        return seq_sent_scores
        #(batch_size, seq_len, sent_count)

    def forward(self, doc_emb, encoder_output_sents_vec, sel_sent_idxs, sel_sent_masks, \
            group_idxs, candi_sent_masks, is_test, raw_sent_embs, sel_sent_hit_map=None):
        #during training seq2seq, sel_sent_idxs is tgt_seq_idxs
        #during inference, sel_sent_idxs is the previous output idxs
        if is_test:
            return self.translate(doc_emb, encoder_output_sents_vec, sel_sent_idxs, sel_sent_masks, \
            candi_sent_masks, raw_sent_embs)
        else:
            return self.decode(doc_emb, encoder_output_sents_vec, sel_sent_idxs, \
            candi_sent_masks, raw_sent_embs)



