import os
import heapq

import numpy as np
import torch
from tensorboardX import SummaryWriter
import collections as coll

import distributed
import json
# import onmt
from models.reporter import ReportMgr
from models.stats import Statistics
import models.data_util as du
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str
import torch.nn.functional as F
import random
from prepro.utils import _get_word_ngrams
from prepro.data_builder import cal_rouge
from models import data_loader
from models.data_loader import load_dataset

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model,
                  optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optim,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.device = "cpu" if self.n_gpu == 0 else "cuda"

        self.loss = torch.nn.BCELoss(reduction='none')
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.softmax = torch.nn.Softmax(dim = -1)
        self.logsoftmax = torch.nn.LogSoftmax(dim = -1)
        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step =  self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        neg_valid_loss = [] # minheap, minum value at top
        heapq.heapify(neg_valid_loss) # use neg loss to find top 3 largest neg loss

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        #select_counts = np.random.choice(range(3), train_steps + 1)
        cur_epoch = 0
        train_iter = train_iter_fct()
        #logger.info('Current Epoch:%d' % cur_epoch)
        #logger.info('maxEpoch:%d' % self.args.max_epoch)

        #while step <= train_steps:
        while cur_epoch < self.args.max_epoch:
            reduce_counter = 0
            logger.info('Current Epoch:%d' % cur_epoch)
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    # from batch.labels, add selected sent index to batch
                    # after teacher forcing, use model selected sentences
                    # or infer scores of batch and get selected sent index
                    # then add selected sent index to the batch

                    true_batchs.append(batch)
                    #normalization += batch.batch_size ##loss normalized wrong
                    normalization = batch.batch_size ##loss recorded correspond to each minibatch
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            valid_iter =data_loader.Dataloader(self.args, load_dataset(self.args, 'valid', shuffle=False),
                                                          self.args.batch_size * 10, self.device,
                                                          shuffle=False, is_test=True)
                            #batch_size train: 3000, test: 60000
                            stats = self.validate(valid_iter, step, self.args.valid_by_rouge)
                            self.model.train() # back to training
                            cur_valid_loss = stats.xent()
                            checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
                            # if len(neg_valid_loss) < self.args.save_model_count:
                            self._save(step)
                            heapq.heappush(neg_valid_loss, (-cur_valid_loss, checkpoint_path))
                            # else:
                            #     if -cur_valid_loss > neg_valid_loss[0][0]:
                            #         heapq.heappush(neg_valid_loss, (-cur_valid_loss, checkpoint_path))
                            #         worse_loss, worse_model = heapq.heappop(neg_valid_loss)
                            #         os.remove(worse_model)
                            #         self._save(step)
                                #else do not save it
                            logger.info('step_%d:%s' % (step, str(neg_valid_loss)))

                        step += 1
                        if step > train_steps:
                            break
            cur_epoch += 1
            train_iter = train_iter_fct()

        return total_stats, neg_valid_loss

    def validate(self, valid_iter, step=0, valid_by_rouge=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src, labels, segs = batch.src, batch.labels, batch.segs
                clss, mask, mask_cls = batch.clss, batch.mask, batch.mask_cls
                #group_idxs, pair_masks = batch.groups, batch.pair_masks
                group_idxs = batch.groups
                soft_labels = batch.soft_labels
                candi_masks = batch.candi_masks
                if valid_by_rouge:
                    src_str, tgt_str = batch.src_str, batch.tgt_str
                    # add negative rouge score as loss to be used as a criterion
                    sel_sent_idxs, sel_sent_masks = self.model.infer_sentences(batch, 3)
                    sel_sent_idxs = sel_sent_idxs.tolist()
                    total_rouge = 0.
                    for i in range(len(sel_sent_idxs)):
                        rouge = du.cal_rouge_doc(src_str[i], tgt_str[i], sel_sent_idxs[i], sel_sent_masks[i])
                        total_rouge += rouge
                    loss = -total_rouge
                else:
                    if self.args.model_name == 'seq':
                        sent_scores, _ = self.model(src, mask, segs, clss, mask_cls, group_idxs,
                                pair_masks, sel_sent_idxs=sel_sent_idxs, sel_sent_masks=sel_sent_masks,
                                candi_sent_masks=candi_masks)
                        #batch, seq_len, sent_count
                        pred = sent_scores.contiguous().view(-1, sent_scores.size(2))
                        gold = batch.label_seq.contiguous().view(-1)
                        if self.args.use_rouge_label:
                            soft_labels = soft_labels.contiguous().view(-1, soft_labels.size(2))
                            #batch*seq_len, sent_count
                            log_prb = F.log_softmax(pred, dim=1)
                            non_pad_mask = gold.ne(-1) # padding value
                            sent_mask = mask_cls.unsqueeze(1).expand(-1,sent_scores.size(1),-1)
                            sent_mask = sent_mask.contiguous().view(-1, sent_scores.size(2))
                            loss = -((soft_labels * log_prb) * sent_mask.float()).sum(dim=1)
                            loss = loss.masked_select(non_pad_mask).sum()  # average later
                        else:
                            loss = F.cross_entropy(pred, gold, ignore_index=-1, reduction='sum')
                    else:
                        sel_sent_idxs, sel_sent_masks = batch.sel_sent_idxs, batch.sel_sent_masks
                        sent_scores, _ = self.model(src, mask, segs, clss, mask_cls, group_idxs, \
                                sel_sent_idxs=sel_sent_idxs,
                                sel_sent_masks=batch.sel_sent_masks, candi_sent_masks=candi_masks, is_test=True,
                                sel_sent_hit_map=batch.hit_map)
                        if self.args.use_rouge_label:
                            labels = soft_labels
                        if self.args.loss == "bce":
                            loss = self.bce_logits_loss(sent_scores, labels.float()) #pointwise
                        elif self.args.loss == "wsoftmax":
                            loss = -self.logsoftmax(sent_scores) * labels.float()
                            #weighted average
                        else:
                            sum_labels = labels.sum(dim=-1).unsqueeze(-1).expand_as(labels)
                            labels = torch.where(sum_labels==0, labels, labels/sum_labels)
                            loss = -self.logsoftmax(sent_scores) * labels.float()

                        #batch_size, max_sent_count
                        loss = (loss*candi_masks.float()).sum()
                        loss = float(loss.cpu().data.numpy())
                batch_stats = Statistics(loss, len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def iter_test(self, test_iter, step, sum_sent_count=3):
        """
           select sentences in each iteration
           given selected sentences, predict the next one
        """
        self.model.eval()
        stats = Statistics()
        #dir_name = os.path.dirname(self.args.result_path)
        base_name = os.path.basename(self.args.result_path)
        #base_dir = os.path.join(dir_name, 'iter_eval')
        base_dir = os.path.dirname(self.args.result_path)

        if (not os.path.exists(base_dir)):
            os.makedirs(base_dir)

        can_path = '%s/%s_step%d_itereval.candidate'%(base_dir, base_name, step)
        gold_path = '%s/%s_step%d_itereval.gold' % (base_dir, base_name, step)

        all_pred_ids, all_gold_ids, all_doc_ids = [], [], []
        all_gold_texts, all_pred_texts = [], []

        with torch.no_grad():
            for batch in test_iter:
                doc_ids = batch.doc_id
                oracle_ids = [set([j for j in seq if j > -1]) for seq in batch.label_seq.tolist()]

                sel_sent_idxs, sel_sent_masks = self.model.infer_sentences(batch, sum_sent_count, stats=stats)
                sel_sent_idxs = sel_sent_idxs.tolist()
                all_pred_ids.extend(sel_sent_idxs)

                for i in range(batch.batch_size):
                    _pred = '<q>'.join([batch.src_str[i][idx].strip() for j, idx in enumerate(sel_sent_idxs[i]) if sel_sent_masks[i][j]])
                    all_pred_texts.append(_pred)
                    all_gold_texts.append(batch.tgt_str[i])
                    all_gold_ids.append(oracle_ids[i])
                    all_doc_ids.append(doc_ids[i])
        macro_precision, micro_precision = self._output_predicted_summaries(
                all_doc_ids, all_pred_ids, all_gold_ids,
                all_pred_texts, all_gold_texts, can_path, gold_path)
        rouge1_arr, rouge2_arr = du.cal_rouge_score(all_pred_texts, all_gold_texts)
        rouge_1, rouge_2 = du.aggregate_rouge(rouge1_arr, rouge2_arr)
        logger.info('[PERF]At step %d: rouge1:%.2f rouge2:%.2f' % (
            step, rouge_1 * 100, rouge_2 * 100))
        if(step!=-1 and self.args.report_precision):
            macro_arr = ["P@%s:%.2f%%" % (i+1, macro_precision[i] * 100) for i in range(3)]
            micro_arr = ["P@%s:%.2f%%" % (i+1, micro_precision[i] * 100) for i in range(3)]
            logger.info('[PERF]MacroPrecision at step %d: %s' % (step, '\t'.join(macro_arr)))
            logger.info('[PERF]MicroPrecision at step %d: %s' % (step, '\t'.join(micro_arr)))
        if(step!=-1 and self.args.report_rouge):
            rouge_str, detail_rouge = test_rouge(self.args.temp_dir, can_path, gold_path, all_doc_ids, show_all=True)
            logger.info('[PERF]Rouges at step %d: %s \n' % (step, rouge_str))
            result_path = '%s_step%d_itereval.rouge' % (self.args.result_path, step)
            if detail_rouge is not None:
                du.output_rouge_file(result_path, rouge1_arr, rouge2_arr, detail_rouge, all_doc_ids)
        self._report_step(0, step, valid_stats=stats)

        return stats


    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        base_dir = os.path.dirname(self.args.result_path)
        if (not os.path.exists(base_dir)):
            os.makedirs(base_dir)

        can_path = '%s_step%d_initial.candidate'%(self.args.result_path,step)
        gold_path = '%s_step%d_initial.gold' % (self.args.result_path, step)

        all_pred_ids, all_gold_ids, all_doc_ids = [], [], []
        all_gold_texts, all_pred_texts = [], []

        with torch.no_grad():
            for batch in test_iter:
                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls
                doc_ids = batch.doc_id
                group_idxs = batch.groups

                oracle_ids = [set([j for j in seq if j > -1]) for seq in batch.label_seq.tolist()]

                if (cal_lead):
                    selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                elif (cal_oracle):
                    selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                    range(batch.batch_size)]
                else:
                    sent_scores, mask = self.model(src, mask, segs, clss, mask_cls, group_idxs, candi_sent_masks=mask_cls, is_test=True)
                    #selected sentences in candi_masks can be set to 0
                    loss = -self.logsoftmax(sent_scores) * labels.float() #batch_size, max_sent_count
                    loss = (loss*mask.float()).sum()

                    batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                    stats.update(batch_stats)

                    sent_scores[mask==False] = float('-inf')
                    # give a cap 1 to sentscores, so no need to add 1000
                    sent_scores = sent_scores.cpu().data.numpy()
                    selected_ids = np.argsort(-sent_scores, 1)
                for i, idx in enumerate(selected_ids):
                    _pred = []
                    _pred_ids = []
                    if(len(batch.src_str[i])==0):
                        continue
                    for j in selected_ids[i][:len(batch.src_str[i])]:
                        if(j>=len( batch.src_str[i])):
                            continue
                        candidate = batch.src_str[i][j].strip()
                        if(self.args.block_trigram):
                            if(not _block_tri(candidate,_pred)):
                                _pred.append(candidate)
                                _pred_ids.append(j)
                        else:
                            _pred.append(candidate)
                            _pred_ids.append(j)

                        if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 2):
                          break

                    _pred = '<q>'.join(_pred)
                    if(self.args.recall_eval):
                        _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                    all_pred_texts.append(_pred)
                    all_pred_ids.append(_pred_ids)
                    all_gold_texts.append(batch.tgt_str[i])
                    all_gold_ids.append(oracle_ids[i])
                    all_doc_ids.append(doc_ids[i])
        macro_precision, micro_precision = self._output_predicted_summaries(
                all_doc_ids, all_pred_ids, all_gold_ids,
                all_pred_texts, all_gold_texts, can_path, gold_path)
        rouge1_arr, rouge2_arr = du.cal_rouge_score(all_pred_texts, all_gold_texts)
        rouge_1, rouge_2 = du.aggregate_rouge(rouge1_arr, rouge2_arr)
        logger.info('[PERF]At step %d: rouge1:%.2f rouge2:%.2f' % (
            step, rouge_1 * 100, rouge_2 * 100))

        if(step!=-1 and self.args.report_precision):
            macro_arr = ["P@%s:%.2f%%" % (i+1, macro_precision[i] * 100) for i in range(3)]
            micro_arr = ["P@%s:%.2f%%" % (i+1, micro_precision[i] * 100) for i in range(3)]
            logger.info('[PERF]MacroPrecision at step %d: %s' % (step, '\t'.join(macro_arr)))
            logger.info('[PERF]MicroPrecision at step %d: %s' % (step, '\t'.join(micro_arr)))

        if(step!=-1 and self.args.report_rouge):
            rouge_str, detail_rouge = test_rouge(self.args.temp_dir, can_path, gold_path, all_doc_ids, show_all=True)
            logger.info('[PERF]Rouges at step %d: %s \n' % (step, rouge_str))
            result_path = '%s_step%d_initial.rouge' % (self.args.result_path, step)
            if detail_rouge is not None:
                du.output_rouge_file(result_path, rouge1_arr, rouge2_arr, detail_rouge, all_doc_ids)
        self._report_step(0, step, valid_stats=stats)

        return stats

    def _output_predicted_summaries(self, all_doc_ids, all_pred_ids, all_gold_ids,
            all_pred_texts, all_gold_texts, can_path, gold_path):
        result_path = can_path.replace('candidate', 'precs')
        micro_precision, macro_precision, valid_count = np.zeros(3), np.zeros(3), 0
        precision_arr = []
        with open(can_path, 'w') as save_pred, open(gold_path, 'w') as save_gold, \
                open(result_path, 'w') as save_prec:
            if not all_doc_ids[0]: #there is no doc_id, use the original way
                for i in range(len(all_gold_texts)):
                    save_gold.write(all_gold_texts[i].strip()+'\n')
                    save_pred.write(all_pred_texts[i].strip()+'\n')
                    hits = np.asarray([1.0 if x in all_gold_ids[i] else 0 for x in all_pred_ids[i]] + (3-len(all_pred_ids[i])) * [0])
                    if len(all_pred_ids[i]) < 3:
                        print('too short', all_pred_ids[i], all_gold_ids[i])

                    #print(hits)
                    precision = np.asarray([sum(hits[:j])/float(j) for j in range(1, len(hits)+1)])
                    #print(precision)
                    micro_precision += precision
                micro_precision /= len(all_pred_ids)
                macro_precision = micro_precision
            else:
                doc_truth_dic = dict()
                doc_id_order = [] #real id list
                for i in range(len(all_pred_texts)):
                    if all_doc_ids[i] in doc_truth_dic:
                        continue
                    save_pred.write(all_pred_texts[i].strip()+'\n')
                    doc_truth_dic[all_doc_ids[i]] = coll.defaultdict(list) #[[],[]]
                    doc_id_order.append(all_doc_ids[i])

                for i in range(len(all_gold_texts)):
                    assert all_doc_ids[i] in doc_truth_dic
                    #the target sentences might be too short (less than 5) to be included in the source text.
                    #the labels are remapped after filtering the sentences, so labels might be emtpy, target sentences still exist
                    if len(all_pred_ids[i]) == 0:
                        print(all_gold_ids[i])
                        print(all_doc_ids[i])
                        print(all_pred_ids[i])
                        #just discard the it. e.g. 31835.docx
                        continue
                    valid_count += 1
                    #precision = sum([1.0 for x in all_pred_ids[i] if x in all_gold_ids[i]])/len(all_pred_ids[i])
                    hits = np.asarray([1.0 if x in all_gold_ids[i] else 0 for x in all_pred_ids[i]] + (3-len(all_pred_ids[i])) * [0])
                    precision = np.asarray([sum(hits[:j])/float(j) for j in range(1, len(hits)+1)])
                    micro_precision += precision
                    doc_truth_dic[all_doc_ids[i]]['gold'].append(all_gold_texts[i].strip())
                    doc_truth_dic[all_doc_ids[i]]['prec'].append(precision)
                    doc_truth_dic[all_doc_ids[i]]['pred_id'].append([str(x) for x in all_pred_ids[i]])
                    doc_truth_dic[all_doc_ids[i]]['gold_id'].append([str(x) for x in all_gold_ids[i]])
                save_prec.write("DocId\tP@1\tP@2\tP@3\tPredIDs\tGoldIDs\n")
                for doc_i in doc_id_order:
                    save_gold.write("{}\n".format("<JG>".join(doc_truth_dic[doc_i]['gold'])))
                    ref_count = min(len(doc_truth_dic[doc_i]['gold']), 1)
                    # cur_avg_precs = sum(doc_truth_dic[doc_i]['prec']) / ref_count
                    try:
                      cur_avg_precs = sum(doc_truth_dic[doc_i]['prec']) / ref_count
                    except:
                      pass
                    #print(doc_truth_dic[doc_i]['gold_id'])
                    #print(doc_truth_dic[doc_i]['pred_id'])

                    print()
                    gold_ids_str = ";".join([",".join(x) for x in doc_truth_dic[doc_i]['gold_id']])
                    pred_ids_str = ";".join([",".join(x) for x in doc_truth_dic[doc_i]['pred_id']])
                    save_prec.write("{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:s}\t{:s}\n".format(
                        str(doc_i), cur_avg_precs[0], cur_avg_precs[1], cur_avg_precs[2], pred_ids_str, gold_ids_str))
                    macro_precision += cur_avg_precs
                macro_precision /= len(doc_id_order)
                micro_precision /= valid_count
        return macro_precision, micro_precision


    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls
            group_idxs = batch.groups
            #they need to have these two attributes
            sel_sent_idxs = batch.sel_sent_idxs
            sel_sent_masks = batch.sel_sent_masks
            candi_masks = batch.candi_masks
            #pair_masks = batch.pair_masks
            src_str, tgt_str = batch.src_str, batch.tgt_str
            soft_labels = batch.soft_labels

            if self.args.model_name == 'seq':
                sent_scores, _ = self.model(src, mask, segs, clss, mask_cls, group_idxs,
                        sel_sent_idxs=sel_sent_idxs, sel_sent_masks=sel_sent_masks,
                        candi_sent_masks=candi_masks)
                #batch, seq_len, sent_count
                pred = sent_scores.contiguous().view(-1, sent_scores.size(2))
                gold = batch.label_seq.contiguous().view(-1)
                if self.args.use_rouge_label:
                    soft_labels = soft_labels.contiguous().view(-1, soft_labels.size(2))
                    #batch*seq_len, sent_count
                    log_prb = F.log_softmax(pred, dim=1)
                    non_pad_mask = gold.ne(-1) # padding value
                    sent_mask = mask_cls.unsqueeze(1).expand(-1,sent_scores.size(1),-1)
                    sent_mask = sent_mask.contiguous().view(-1, sent_scores.size(2))
                    loss = -((soft_labels * log_prb) * sent_mask.float()).sum(dim=1)
                    loss = loss.masked_select(non_pad_mask).sum()  # average later
                else:
                    loss = F.cross_entropy(pred, gold, ignore_index=-1, reduction='sum')
            else:
                sent_scores, _ = self.model(src, mask, segs, clss, mask_cls, group_idxs,
                        sel_sent_idxs=sel_sent_idxs, sel_sent_masks=sel_sent_masks,
                        candi_sent_masks=candi_masks,
                        sel_sent_hit_map=batch.hit_map)
                if self.args.use_rouge_label:
                    labels = soft_labels
                if self.args.loss == "bce":
                    loss = self.bce_logits_loss(sent_scores, labels.float()) #pointwise
                elif self.args.loss == "wsoftmax":
                    loss = -self.logsoftmax(sent_scores) * labels.float()

                #batch_size, max_sent_count
                loss = (loss*candi_masks.float()).sum()
                #print("loss_sum", loss)

            (loss/loss.numel()).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)
            #print([p for p in self.model.parameters() if p.requires_grad])

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
