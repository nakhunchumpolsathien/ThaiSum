import math

import torch
import torch.nn as nn

from models.encoder import PositionalEncoding
from models.neural import MultiHeadedAttention, PositionwiseFeedForward

def get_subsequent_mask(seq_emb):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, emb_dim = seq_emb.size()
    subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=seq_emb.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.enc_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, dec_input, enc_output, self_attn_mask,
            dec_enc_attn_mask, self_attn_subseq_mask=None):
        if (iter != 0):
            input_norm = self.layer_norm1(dec_input)
        else:
            input_norm = dec_input

        self_attn_mask = self_attn_mask.unsqueeze(1)
        #batch_size, 1, tgt_len
        if self_attn_subseq_mask is not None:
            #batch_size, tgt_len, tgt_len
            self_attn_mask = self_attn_mask.expand_as(self_attn_subseq_mask)
            self_attn_mask = (self_attn_mask + self_attn_subseq_mask).gt(0)


        dec_enc_attn_mask = dec_enc_attn_mask.unsqueeze(1)
        dec_output = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=self_attn_mask)
        dec_output = self.dropout(dec_output) + dec_input
        dec_output = self.layer_norm2(dec_output)
        context = self.enc_attn(enc_output, enc_output, dec_output,
                                 mask=dec_enc_attn_mask)
        out = self.dropout(context) + dec_output
        return self.feed_forward(out)

class TransformerDecoder(nn.Module):
    '''Only apply to when groupsize == 1'''
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0, use_pos_emb=False):
        super(TransformerDecoder, self).__init__()
        self.use_pos_emb = use_pos_emb
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        if self.use_pos_emb:
            self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt_emb, tgt_masks, \
            encoder_output, encoder_masks, subseq_mask=None):
        '''
        tgt_emb: embeddings of target sequence
        tgt_masks: masks (both mask subsequent sentences and padded sentences)
        encoder_output  : batch_size, max_group_count, emb_dim
                    a group of sentences for comparison and scoring
        encoder_masks : batch_size, max_group_count
        decoder_output: batch_size, output_sequece, emb_dim
        '''
        #batch_size, n_sents, emb_dim = encoder_output.size()
        n_tgt_sents = tgt_emb.size(1)
        #print("encoder_output", encoder_output)
        encoder_output = encoder_output.squeeze(2) #encoder_output
        decoder_output = tgt_emb
        if self.use_pos_emb:
            pos_emb = self.pos_emb.pe[:, :n_tgt_sents]
            decoder_output = tgt_emb + pos_emb
        for i in range(self.num_inter_layers):
            decoder_output = self.transformer_inter[i](i, decoder_output, encoder_output,
                    1-tgt_masks.byte(), 1-encoder_masks.byte(), subseq_mask)
        #batch_size, sel_sent_count, emb_dim
        decoder_output = self.layer_norm(decoder_output)
        return decoder_output

