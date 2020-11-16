import gc
import glob
import random
import numpy as np
import torch.nn.functional as F

import torch

from others.logging import logger
import models.data_util as du


class Batch(object):
    def _pad(self, data, pad_id, fix_width=-1):
        width = max(len(d) for d in data)
        if (fix_width > 0):
            width = max(width, fix_width)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, args, data=None, device=None,
            is_test=False):
        """Create a Batch from a list of examples."""
        """rand permutation"""
        """or all the combinations (too slow to try)"""
        #(selected sentences kept as candidates with labels set to 0 or removed from candidates)
        self.args = args
        self.is_empty = False
        self.device = device
        self.default_pad_id = -1
        ngram_segs = [int(x) for x in self.args.ngram_seg_count.split(',')]

        if data is not None:
            sel_sent_idxs, sel_sent_masks = None, None
            groups, soft_labels, hit_map = None, None, None
            self.batch_size = len(data)

            pre_src = [x['src'] for x in data] # a sequence of token id converted with Bert tokenizer
            pre_labels = [x['labels'] for x in data]
            pre_segs = [x['segs'] for x in data]
            pre_clss = [x['clss'] for x in data]
            #print(pre_labels)
            src_str = [x['src_txt'] for x in data] #batch of list of sentence text
            tgt_str = [x['tgt_txt'] for x in data] #batch of list of text concatnated with "<q>"

            label_seq = du.get_label_orders(src_str, tgt_str, pre_labels)
            #print(label_seq)
            label_seq = torch.tensor(self._pad(label_seq, -1, fix_width=self.args.max_label_sent_count)).to(device)
            #in some corner case, none of the doc in a batch has positive labels.

            src = torch.tensor(self._pad(pre_src, 0)).to(device)
            labels = torch.tensor(self._pad(pre_labels, -1)).to(device)
            segs = torch.tensor(self._pad(pre_segs, 0)).to(device)
            mask = 1 - (src == 0) #mask of sequence of tokens  
            mask = mask.to(device)
            clss = torch.tensor(self._pad(pre_clss, -1)).to(device)
            mask_cls = 1 - (clss == -1)
            clss[clss == -1] = 0 #mask of sequence of sentences
            candi_masks = mask_cls.clone().detach()
            if self.args.model_name == 'seq':
                sel_sent_idxs = label_seq.clone().detach()
                if random.random() > self.args.rand_input_thre:
                    du.random_replace_gold(sel_sent_idxs, mask_cls)
            else:
                batch_indices = [list(range(len(sent))) for sent in pre_clss]
                groups = torch.tensor(self._pad(batch_indices, -1)).to(device)

            src_str = [src_str[i][:len(pre_clss[i])] for i in range(self.batch_size)]
            if self.args.model_name == "ctx":
                select_count = np.random.choice(np.arange(1,3), p=[0.5, 0.5])
                if select_count > 0:
                    sel_sent_idxs, sel_sent_masks = du.get_selected_sentences(pre_labels, \
                            select_count, pad_id = self.default_pad_id)
                    sel_sent_idxs = torch.tensor(sel_sent_idxs).to(device)
                    sel_sent_masks = torch.tensor(sel_sent_masks).to(device)
                    if random.random() > self.args.rand_input_thre:
                        du.random_replace_gold(sel_sent_idxs, mask_cls, disregard_last=False)

                    sel_sent_label = 0
                    du.set_selected_sent_to_value(labels, sel_sent_idxs, sel_sent_masks, sel_sent_label)
                    du.set_selected_sent_to_value(candi_masks, sel_sent_idxs, sel_sent_masks, False)
                    hit_map = du.get_hit_ngram(src_str, sel_sent_idxs, sel_sent_masks, ngram_segs, device=device)

            if self.args.use_rouge_label:
                if self.args.model_name == 'seq':
                    soft_labels = du.get_rouge_distr_each_step(src_str, tgt_str, label_seq)
                    soft_labels = torch.tensor(soft_labels).to(device)
                    non_valid_idxs = soft_labels.ne(float('-inf')).sum(dim=-1).eq(0)
                    soft_labels = F.softmax(soft_labels * self.args.temperature, dim=-1)
                    soft_labels[non_valid_idxs] = 0.
                else:
                    soft_labels = labels
                    if sel_sent_idxs is not None:
                        if "greedy" in self.args.label_format:
                            greedy_labels = du.get_greedy_labels(src_str, tgt_str, sel_sent_idxs)
                            soft_labels = torch.tensor(self._pad(greedy_labels, 0)).to(device)
                        else:
                            soft_labels = du.get_soft_labels(src_str, tgt_str, sel_sent_idxs)
                            soft_labels = torch.tensor(self._pad(soft_labels, float('-inf') )).to(device)
                            non_valid_idxs = soft_labels.ne(float('-inf')).sum(dim=-1).eq(0)
                            soft_labels = F.softmax(soft_labels * self.args.temperature, dim=-1)
                            soft_labels[non_valid_idxs] = 0.
                            #use the original label for the case where there is no selected sentences
                            empty_sel_idxs = sel_sent_masks.sum(dim=-1).eq(0)
                            soft_labels[empty_sel_idxs] =  labels[empty_sel_idxs].float()

            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src)
            setattr(self, 'labels', labels)
            setattr(self, 'label_seq', label_seq)
            setattr(self, 'segs', segs)
            setattr(self, 'mask', mask)
            setattr(self, 'src_str', src_str)
            setattr(self, 'tgt_str', tgt_str)
            setattr(self, 'soft_labels', soft_labels)
            setattr(self, 'sel_sent_idxs', sel_sent_idxs)
            setattr(self, 'sel_sent_masks', sel_sent_masks)
            setattr(self, 'hit_map', hit_map)
            setattr(self, 'groups', groups)
            setattr(self, 'candi_masks', candi_masks)

            if (is_test):
                #for case analysis, and multipple reference evaluation
                doc_id = [x['doc_id'] for x in data]
                setattr(self, 'doc_id', doc_id)

    def __len__(self):
        return self.batch_size

def batch(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
    if minibatch:
        yield minibatch


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    #logger.info(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt')
    #logger.info('pts: {}'.format(pts))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        #pt = args.bert_data_path + '.' + corpus_type + '.pt'
        pt = args.bert_data_path + '.' + corpus_type + '.bert.pt' #revised by Keping
        yield _lazy_dataset_loader(pt, corpus_type)


def simple_batch_size_fn(new, count):
    #src, labels = new[0], new[1]
    src, labels = new['src'], new['labels']
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(src))
    #len(src) is the token count in the document
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)

        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, is_test=self.is_test,
            shuffle=self.shuffle)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size,  device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle
        logger.info('input_batch_size:%s' %
                    (self.batch_size))

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        src = ex['src']
        if('labels' in ex):
            labels = ex['labels']
        else:
            labels = ex['src_sent_labels']

        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']
        doc_id = ex.get('doc_id', None)

        return {'src':src,'labels':labels,'segs':segs, \
                'clss':clss, 'doc_id':doc_id, 'src_txt':src_txt, 'tgt_txt':tgt_txt}

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            #the number of tokens so far (max_token_len * size of minibatch)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 50):
            p_batch = sorted(buffer, key=lambda x: len(x['clss']))
            #sort according to the number of sentences
            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            #convert a iteration to list and shuffle the list.
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                # added by Keping
                if not minibatch:
                    continue

                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(self.args, minibatch, self.device, self.is_test)
                if batch.is_empty:
                    continue
                yield batch

            return

