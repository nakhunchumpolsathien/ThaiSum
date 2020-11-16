import torch
import numpy as np
import torch.nn.functional as F
from others.logging import logger
import random
import re
import collections as coll
from prepro.utils import _get_word_ngrams

def get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def block_tri(c, p):
    tri_c = get_ngrams(3, c.split())
    for s in p:
        tri_s = get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s))>0:
            return True
    return False

def union_dic(dic_list):
    #assert len(dic_list) > 0
    if len(dic_list) == 0:
        return coll.defaultdict(int)

    sum_dic = dic_list[0].copy()
    for dic in dic_list[1:]:
        for key in dic:
            sum_dic[key] += dic[key]
    return sum_dic

def cal_rouge(evaluated_ngrams, reference_ngrams):
    #ngrams: count
    ref_count = sum(reference_ngrams.values())
    eval_count = sum(evaluated_ngrams.values())
    match_count = 0
    for k, v in evaluated_ngrams.items():
        if k in reference_ngrams:
            match_count += min(v, reference_ngrams[k])
    precision = match_count / eval_count if eval_count != 0 else 0.
    recall = match_count / ref_count if ref_count != 0 else 0.
    f1_score = 0. if (precision == 0 or recall == 0) else 2 * precision * recall / (precision + recall)

    return {"f": f1_score, "p": precision, "r": recall}


def rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)

def cal_rouge_score(pred_list, truth_list):
    # one entry of src, tgt and selected sentences
    rouge1_arr, rouge2_arr = [], []
    for pred_txt, tgt_txt in zip(pred_list, truth_list):
        gold = tgt_txt.replace("<q>", " ").split()
        reference_1grams = _get_word_ngrams(1, [gold], do_count=True)
        reference_2grams = _get_word_ngrams(2, [gold], do_count=True)

        candi = pred_txt.replace("<q>", " ").split()
        candidates_1 = _get_word_ngrams(1, [candi], do_count=True)
        candidates_2 = _get_word_ngrams(2, [candi], do_count=True)

        rouge_1 =  cal_rouge(candidates_1, reference_1grams)
        rouge1_arr.append(rouge_1)
        rouge_2 = cal_rouge(candidates_2, reference_2grams)
        rouge2_arr.append(rouge_2)
    return rouge1_arr, rouge2_arr

def aggregate_rouge(rouge1_arr, rouge2_arr, metric='f'):
    rouge1 = [x[metric] for x in rouge1_arr]
    rouge2 = [x[metric] for x in rouge2_arr]
    rouge1 = sum(rouge1) / len(rouge1)
    rouge2 = sum(rouge2) / len(rouge2)
    return rouge1, rouge2

def get_hit_ngram(batch_src_txt, batch_c_idxs, batch_sel_idxs_masks, \
        ngram_seg_count=[20,20,20], discrete=True, device='cuda'):
    #count or ratio
    batch_sent_hit_gram = []
    max_sent_count = max([len(batch_src_txt[i]) for i in range(len(batch_src_txt))])
    for idx, seg_count in enumerate(ngram_seg_count):
        batch_hit_gram = []
        for b_i in range(len(batch_src_txt)):
            src_txt = batch_src_txt[b_i]
            c_idxs = batch_c_idxs[b_i]
            c_masks = batch_sel_idxs_masks[b_i]
            gram_hit = get_hit_ngram_one_entry(src_txt, c_idxs, c_masks, max_sent_count, idx+1, seg_count, discrete)
            batch_hit_gram.append(gram_hit)
        batch_sent_hit_gram.append(torch.tensor(batch_hit_gram).to(device))
    return batch_sent_hit_gram

def get_hit_ngram_one_entry(src_txt, c_idxs, c_masks, max_sent_count, gram_n, gram_seg_count, discrete, do_ratio=True):
    sel_sent_ngram = set()
    for idx, valid in zip(c_idxs, c_masks):
        if not valid:
            continue
        sel_sent_ngram = sel_sent_ngram.union(get_ngrams(gram_n, src_txt[idx].split()))

    all_sent_hit_ngram = []
    for i in range(max_sent_count):
        #sentence count
        hit_ngram = [0] * gram_seg_count
        if i >= len(src_txt):
            all_sent_hit_ngram.append(hit_ngram)
            continue
        cur_ngram_set = get_ngrams(gram_n, src_txt[i].split())
        hit_count = len(cur_ngram_set.intersection(sel_sent_ngram))
        if do_ratio:
            try:
              hit_ratio = float(hit_count) / len(cur_ngram_set)
              idx = int(hit_ratio / (1.0/gram_seg_count))               
            except ZeroDivisionError:
              hit_ratio = 0
              idx = int(hit_ratio / (1.0/gram_seg_count))               

            hit_ngram[min(idx, gram_seg_count-1)] = 1. if discrete else hit_ratio
        else:
            hit_ngram[min(hit_count,gram_seg_count-1)] = 1. if discrete else hit_count
        all_sent_hit_ngram.append(hit_ngram)
    return all_sent_hit_ngram


def cal_rouge_doc(src_txt, tgt_txt, c_idxs, c_masks=None):
    # one entry of src, tgt and selected sentences
    abstract = tgt_txt.replace("<q>", " ").split()
    reference_1grams = _get_word_ngrams(1, [abstract], do_count=True)
    reference_2grams = _get_word_ngrams(2, [abstract], do_count=True)

    #the following two are same
    if c_masks is None:
        sents = [src_txt[i].split() for i in c_idxs if i > -1]
    else:
        sents = [src_txt[idx].split() for i,idx in enumerate(c_idxs) if c_masks[i]]

    #-1 is padding id
    evaluated_1grams = [_get_word_ngrams(1, [sent], do_count=True) for sent in sents]
    evaluated_2grams = [_get_word_ngrams(2, [sent], do_count=True) for sent in sents]
    #candidates_1 = set.union(*map(set, evaluated_1grams))
    #candidates_2 = set.union(*map(set, evaluated_2grams))
    candidates_1 = union_dic(evaluated_1grams)
    candidates_2 = union_dic(evaluated_2grams)

    rouge_1 = cal_rouge(candidates_1, reference_1grams)
    rouge_2 = cal_rouge(candidates_2, reference_2grams)
    #print(rouge_1['p'], rouge_1['r'], rouge_1['f'])
    #print(rouge_2['p'], rouge_2['r'], rouge_2['f'])
    rouge_score = rouge_1['f'] + rouge_2['f']
    return rouge_score

def cal_rouge_gain(src_txt, tgt_txt, c_idxs, rouge2_ratio=0.5, do_norm=True):
    '''
    sents: all the sentences
    abstract: target ground truth text
    c_idxs: collection of selected indexs
    '''
    sents = [sent.split() for sent in src_txt]
    abstract = tgt_txt.replace("<q>", " ").split()
    evaluated_1grams = [_get_word_ngrams(1, [sent], do_count=True) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract], do_count=True)
    evaluated_2grams = [_get_word_ngrams(2, [sent], do_count=True) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract], do_count=True)
    #print(reference_1grams, reference_2grams)
    #print(evaluated_1grams, evaluated_2grams)
    if c_idxs is None or len(c_idxs) == 0:
        rouge_score = 0.0
        candidates_1 = coll.defaultdict(int)
        candidates_2 = coll.defaultdict(int)
    else:
        candidates_1 = [evaluated_1grams[idx] for idx in c_idxs if idx > -1] #fix a little bug
        #candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx] for idx in c_idxs if idx > -1]
        #candidates_2 = set.union(*map(set, candidates_2))
        candidates_1 = union_dic(candidates_1)
        candidates_2 = union_dic(candidates_2)
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        rouge_score = rouge_1 * (1-rouge2_ratio) + rouge_2 * rouge2_ratio
    #print(rouge_score)
    min_gain = float('inf')
    max_gain = float('-inf')
    rouge_gain = [float('-inf')] * len(sents)
    for i in range(len(sents)):
        if c_idxs is not None and i in c_idxs:
            continue
        unigrams = union_dic([candidates_1, evaluated_1grams[i]])
        bigrams = union_dic([candidates_2, evaluated_2grams[i]])
        #unigrams = candidates_1.union(set(evaluated_1grams[i]))
        #bigrams = candidates_2.union(set(evaluated_2grams[i]))
        #print(unigrams, bigrams)
        cur_rouge_1 = cal_rouge(unigrams, reference_1grams)['f']
        cur_rouge_2 = cal_rouge(bigrams, reference_2grams)['f']
        #print(cur_rouge_1, cur_rouge_2)
        cur_rouge_score = cur_rouge_1 * (1-rouge2_ratio) + cur_rouge_2 * rouge2_ratio
        #cur_rouge_score = cur_rouge_1 + cur_rouge_2
        cur_gain = cur_rouge_score - rouge_score
        rouge_gain[i] = cur_gain
        max_gain = max(max_gain, cur_gain)
        min_gain = min(min_gain, cur_gain)
    gain_gap = max_gain - min_gain
    #print(gain_gap)
    rouge_gain = np.asarray(rouge_gain)
    #print(rouge_gain)
    if do_norm and gain_gap > 0:
        rouge_gain = (rouge_gain - min_gain) / gain_gap
        #same after softmax
    #do clip
    #rouge_gain[rouge_gain<0] = float('-inf')

    return rouge_gain

def get_soft_labels(batch_src_sents, batch_tgt_str, batch_sel_idxs): #, temperature=20.0):
    '''batch_src_sents: batch of list of source sentences containing tokens
       batch_tgt_str: batch of ground truth summary
       batch_sel_sents: batch of list of selected sentences containing tokens
       return softlabels #all the selected sentences get 0 gain, the scores of the others are normalized to sum to 1
    '''
    # get rouge of selected sentences according to ground truth.
    # for each candidate sentence, get the rouge gain of each sentence.
    batch_size = len(batch_src_sents)
    batch_rouge_gain = []
    for i in range(batch_size):
        cur_sel_idxs = None
        if batch_sel_idxs is not None:
            cur_sel_idxs = batch_sel_idxs[i]
        rouge_gain = cal_rouge_gain(batch_src_sents[i], batch_tgt_str[i], cur_sel_idxs)
        batch_rouge_gain.append(rouge_gain.tolist())
    #batch_rouge_gain = torch.tensor(batch_rouge_gain)
    #batch_rouge_gain = F.softmax(batch_rouge_gain * temperature, dim=-1)
    return batch_rouge_gain

def get_greedy_labels(batch_src_sents, batch_tgt_str, batch_sel_idxs):
    '''batch_src_sents: batch of list of source sentences containing tokens
       batch_tgt_str: batch of ground truth summary
       batch_sel_sents: batch of list of selected sentences containing tokens
       return softlabels #all the selected sentences get 0 gain, the scores of the others are normalized to sum to 1
    '''
    # get rouge of selected sentences according to ground truth.
    # for each candidate sentence, get the rouge gain of each sentence.
    batch_size = len(batch_src_sents)
    batch_greedy_label = []
    for i in range(batch_size):
        cur_sel_idxs = None
        if batch_sel_idxs is not None:
            cur_sel_idxs = batch_sel_idxs[i]

        selected_idxs = greedy_selection_given_context(batch_src_sents[i], batch_tgt_str[i], cur_sel_idxs)
        cur_labels = [0] * len(batch_src_sents[i])
        for sid in selected_idxs:
            cur_labels[sid] = 1

        batch_greedy_label.append(cur_labels)
    return batch_greedy_label

def greedy_selection_given_context(src_txt, tgt_txt, c_idxs, summary_size = 3):
    sents = [sent.split() for sent in src_txt]
    abstract = tgt_txt.replace("<q>", " ").split()
    evaluated_1grams = [_get_word_ngrams(1, [sent], do_count=True) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract], do_count=True)
    evaluated_2grams = [_get_word_ngrams(2, [sent], do_count=True) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract], do_count=True)
    #print(reference_1grams, reference_2grams)
    #print(evaluated_1grams, evaluated_2grams)

    max_rouge = 0.0
    selected = []
    if c_idxs is not None:
        for cid in c_idxs:
            if cid >-1:
                selected.append(cid)
    sel_count = len(selected)
    summary_sent_count = summary_size - len(selected)
    #print(sel_count)
    for s in range(summary_sent_count):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            #candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            #candidates_2 = set.union(*map(set, candidates_2))
            candidates_1 = union_dic(candidates_1)
            candidates_2 = union_dic(candidates_2)
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected[sel_count:]
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return selected[sel_count:]



def get_rouge_distr_each_step(batch_src_sents, batch_tgt_str, tgt_seq_idxs, max_seq_len = 3):
    batch_size = len(batch_src_sents)
    if max_seq_len is None:
        max_seq_len = max([len(seq) for seq in tgt_seq_idxs])
    max_sent_len = max([len(seq) for seq in batch_src_sents])
    batch_rouge_gain = []
    for i in range(batch_size):
        rouge_dist_per_step = []
        candidates = [[]] + [tgt_seq_idxs[i][:j] for j in range(1,len(tgt_seq_idxs[i]))]
        for cur_sel_idxs in candidates:
            #[], [1],[1,2] for [1,2,3]
            rouge_gain = cal_rouge_gain(batch_src_sents[i], batch_tgt_str[i], cur_sel_idxs)
            rouge_gain = rouge_gain.tolist() + [float('-inf')] * (max_sent_len - len(rouge_gain))
            rouge_dist_per_step.append(rouge_gain)
        rouge_dist_per_step += [[float('-inf')] * max_sent_len]* (max_seq_len - len(candidates))
        batch_rouge_gain.append(rouge_dist_per_step)
    #batch_rouge_gain = torch.tensor(batch_rouge_gain)
    #batch_rouge_gain = F.softmax(batch_rouge_gain * temperature, dim=-1)
    return batch_rouge_gain

def random_replace_gold(label_seq, mask_cls, disregard_last=True):
    #label_seq: [[0,2,-1],[7,5,3]]
    #mask_cls: [[1,1,1,0,0,0,0],[1,1,1,1,1,1,1]]
    label_count = label_seq.ne(-1).sum(dim=-1)
    batch_size, max_label_count = label_seq.size()
    sent_count = mask_cls.gt(0).sum(dim=-1)
    for i in range(batch_size):
        prev_label_count = label_count[i]
        if not disregard_last:
            prev_label_count += 1
        if prev_label_count > 1:
            sel_count = random.choice(range(1, prev_label_count))
            #print(sel_count)
            #1 or 2
            sel_idxs = random.sample(range(sent_count[i]), sel_count)
            #print(sel_idxs)
            for j in range(sel_count):
                label_seq[i][j] = sel_idxs[j]

    #label_seq has been changed


def get_label_orders(batch_src_sents, batch_tgt_str, batch_labels): #, temperature=20.0):
    '''batch_src_sents: batch of list of source sentences containing tokens
       batch_tgt_str: batch of ground truth summary
       batch_labels: batch of labels such as [0,0,0,1,0] indicating ground truth sentences
       return batch of orders according to max marginal gain [[2,1,0],[6,9]] before padding
    '''
    # get rouge of selected sentences according to ground truth.
    # for each candidate sentence, get the rouge gain of each sentence.
    batch_size = len(batch_src_sents)
    batch_label_seq = []
    for i in range(batch_size):
        label = batch_labels[i]
        rel_idxs = [idx for idx in range(len(label)) if label[idx] == 1]
        selected_idxs = []
        src_txt, tgt_txt = batch_src_sents[i], batch_tgt_str[i]
        sents = [src_txt[x].split() for x in rel_idxs]
        abstract = tgt_txt.replace("<q>", " ").split()
        evaluated_1grams = [_get_word_ngrams(1, [sent], do_count=True) for sent in sents]
        reference_1grams = _get_word_ngrams(1, [abstract], do_count=True)
        evaluated_2grams = [_get_word_ngrams(2, [sent], do_count=True) for sent in sents]
        reference_2grams = _get_word_ngrams(2, [abstract], do_count=True)
        #print('ref', reference_1grams)
        idxs = list(range(len(rel_idxs)))
        #print('eval', evaluated_1grams)
        #print(rel_idxs)
        while len(idxs) > 0:
            cur_id = -1
            cur_max_rouge = float('-inf')
            #cur_max_rouge should be larger than 0, in case of any illegal case, relax it to be possiblely negative
            for r_idx in idxs:
                c = selected_idxs + [r_idx]
                candidates_1 = [evaluated_1grams[idx] for idx in c]
                #candidates_1 = set.union(*map(set, candidates_1))
                candidates_2 = [evaluated_2grams[idx] for idx in c]
                #candidates_2 = set.union(*map(set, candidates_2))
                candidates_1 = union_dic(candidates_1)
                candidates_2 = union_dic(candidates_2)
                rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
                rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
                rouge_score = rouge_1 + rouge_2
                #print(candidates_1)
                #print(r_idx, c, rouge_score, cur_max_rouge)
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = r_idx
                #print(cur_id)
            assert cur_id > -1
            idxs.remove(cur_id)
            selected_idxs += [cur_id]
        label_seq = [rel_idxs[x] for x in selected_idxs]
        batch_label_seq.append(label_seq)
    return batch_label_seq


def set_selected_sent_to_value(labels, sel_idxs, sel_masks, sel_sent_label=-2):
    #only change the idxs where the mask is 1, i.e., -1 is not considered
    if type(labels) is torch.Tensor:
        #labels[torch.arange(len(labels)).unsqueeze(-1), sel_idxs] *= (1-sel_masks).float()
        #change labels of selected idx to 0
        #labels[torch.arange(len(labels)).unsqueeze(-1), sel_idxs] = (1-sel_masks).float()
        #change labels of selected idxs to -1 or other values, we change padded values to -1
        ori_values = labels[torch.arange(len(labels)).unsqueeze(-1), sel_idxs]
        values = torch.tensor(sel_sent_label).to(str(labels.device)).expand_as(sel_idxs)


        labels[torch.arange(len(labels)).unsqueeze(-1), sel_idxs] = torch.where(sel_masks==1, values, ori_values)
    else: # list
        for x in range(len(labels)):
            for i in range(len(sel_idxs[x])):
                if sel_masks[x][i] > 0:
                    labels[x][sel_idxs[x][i]] = sel_sent_label #set the label of selected sentence to 0
    #print(pre_labels)

def get_selected_sentences(labels, selected_count, pad_id = -1, method="truth"):
    # labels #from which step, use 1 sentence, and which step use 2 sentences.
    # the ratio of samples under each context
    # selected from lead 3 or ground truth
    # it can also be from the model prediction.
    # batch_size * selected_count
    assert selected_count < 3 and selected_count > -1
    if selected_count == 0:
        #return torch.tensor([]), torch.tensor([])
        return None, None
    select_idxs = []
    masks = []
    for label in labels:
        if method == "truth":
            rel_idxs = [idx for idx in range(len(label)) if label[idx] == 1]
            if len(rel_idxs) > selected_count:
                sel_idxs = random.sample(rel_idxs, selected_count)
                mask = [1] * selected_count
            elif len(rel_idxs) > 1:
                sel_idxs = random.sample(rel_idxs, len(rel_idxs) - 1)
                sel_idxs += [pad_id]
                mask = [1] * (len(rel_idxs) - 1) + [0] * (selected_count - len(rel_idxs) + 1)
            else:
                #len(rel_idxs) == 1 or 0
                sel_idxs = [pad_id] * selected_count
                mask = [0] * selected_count
                #mask = [1] * len(rel_idxs) + [0] * (selected_count - len(rel_idxs))
        select_idxs.append(sel_idxs)
        masks.append(mask)
    #return torch.tensor(select_idxs), torch.tensor(masks)
    return select_idxs, masks


def compute_metrics(can_path, gold_path):
    all_pred_texts = []
    all_gold_texts = []
    pred_txt_arrs = []
    gold_txt_arrs = []
    with open(can_path) as fcan:
        for line in fcan:
            all_pred_texts.append(line.strip())
            #pred_txt_arrs.append([rouge_clean(x) for x in line.strip().split('<q>')])
    with open(gold_path) as fgold:
        for line in fgold:
            all_gold_texts.append(line.strip())
            #gold_txt_arrs.append(set([rouge_clean(x) for x in line.strip().split('<q>')]))
    rouge1_arr, rouge2_arr = cal_rouge_score(all_pred_texts, all_gold_texts)
    rouge1f = sum([x['f'] for x in rouge1_arr])/len(rouge1_arr)
    rouge2f = sum([x['f'] for x in rouge2_arr])/len(rouge2_arr)

    logger.info("Rouge1:%.2f Rouge2:%.2f" % (rouge1f * 100, rouge2f * 100))
    print("Rouge1:%.2f Rouge2:%.2f" % (rouge1f * 100, rouge2f * 100))
    return rouge1_arr, rouge2_arr

def read_prec_file(prec_file):
    doc_ids = []
    with open(prec_file, 'r') as fp:
        fp.readline()
        for line in fp:
            if 'docId' in line:
                continue
            doc_id = line.strip().split('\t')[0]
            doc_ids.append(doc_id)
    return doc_ids

def output_rouge_file(out_file, rouge1_arr, rouge2_arr, detail_rouge, all_doc_ids):
    '''
    rouge1_arr:[{'f':0.223, 'p':0.234, 'r':0.***}, {...}]
    rouge2_arr: similar as rouge1_arr
        computed with python functions
    detail_rouge: perl version rouge
    '''
    assert detail_rouge is not None
    with open(out_file, "w") as fp:
        fp.write("DocId\t1-F\t1-R\t1-P\t2-F\t2-R\t2-P\tperl-1-F\tperl-1-R\tperl-1-P\tperl-2-F\tperl-2-R\tperl-2-P\tperl-L-F\n")
        for i, doc_id in enumerate(all_doc_ids):
            segs = []
            each_rouge = detail_rouge['individual'][doc_id]
            segs.append(rouge1_arr[i]['f'])
            segs.append(rouge1_arr[i]['r'])
            segs.append(rouge1_arr[i]['p'])
            segs.append(rouge2_arr[i]['f'])
            segs.append(rouge2_arr[i]['r'])
            segs.append(rouge2_arr[i]['p'])
            segs.append(each_rouge['rouge-1-F'])
            segs.append(each_rouge['rouge-1-R'])
            segs.append(each_rouge['rouge-1-P'])
            segs.append(each_rouge['rouge-2-F'])
            segs.append(each_rouge['rouge-2-R'])
            segs.append(each_rouge['rouge-2-P'])
            segs.append(each_rouge['rouge-L-F'])
            seg_str_arr = ['%.2f' % seg for seg in segs]
            fp.write("%s\t%s\n" % (doc_id, '\t'.join(seg_str_arr)))

