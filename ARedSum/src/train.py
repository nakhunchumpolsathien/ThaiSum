#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import heapq

import torch
from pytorch_pretrained_bert import BertConfig

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import Summarizer
from models.trainer import build_trainer
from others.logging import logger, init_logger
from others.utils import test_rouge, rouge_results_to_str
import models.data_util as du

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers','encoder','ff_actv', 'use_interval','rnn_size']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def multi_main(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
            device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()



def run(args, device_id, error_queue):

    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' %gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train(args,device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)



def wait_and_validate(args, device_id):

    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        #for i, cp in enumerate(cp_files):
        for i, cp in enumerate(cp_files[-30:]): #revised by Keping
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args,  device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                #if the model currently under validation is 10 models further from the best model, stop
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test(args,  device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args,  device_id, cp, step)
                    test(args,  device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args,  device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    valid_iter =data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter, step, valid_by_rouge=args.valid_by_rouge)
    return stats.xent()

def test(args, device_id, pt, step):

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    #model.load_cp(checkpoint) #TODO: change it back to strict=True
    model.load_cp(checkpoint, strict=False)
    model.eval()

    trainer = build_trainer(args, device_id, model, None)
    #if False:
    #args.block_trigram = True
    #if not args.only_initial or args.model_name == 'seq':
    if args.model_name == 'base':
        test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                      args.batch_size, device,
                                      shuffle=False, is_test=True)
        trainer.test(test_iter,step)
    else:
        test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
        trainer.iter_test(test_iter,step)
        #for iterative ranker, evaluate both initial ranker and iterative ranker

def baseline(args, cal_lead=False, cal_oracle=False):

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)

    trainer = build_trainer(args, device_id, None, None)
    #
    if (cal_lead):
        trainer.test(test_iter, 0, cal_lead=True)
    elif (cal_oracle):
        trainer.test(test_iter, 0, cal_oracle=True)


def train(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                                 shuffle=True, is_test=False)

    # temp change for reducing gpu memory
    model = Summarizer(args, device, load_pretrained_bert=True)
    #config = BertConfig.from_json_file(args.bert_config_path)
    #model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)

    if args.train_from != '': #train another part from beginning
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        model.load_cp(checkpoint, strict=False)
        # keys can not match
        #optim = model_builder.build_optim(args, model, checkpoint)
        optim = model_builder.build_optim(args, model, None)
        if args.model_name == "ctx" and args.fix_scorer:
            logger.info("fix the saliency scorer")
            #for param in self.bert.model.parameters():
            for param in model.parameters():
                param.requires_grad = False

            if hasattr(model.encoder, "selector") and model.encoder.selector is not None:
                for param in model.encoder.selector.parameters():
                    param.requires_grad = True
            #print([p for p in model.parameters() if p.requires_grad])
    else:
        optim = model_builder.build_optim(args, model, None)

    logger.info(model)
    trainer = build_trainer(args, device_id, model, optim)
    _, neg_valid_loss = trainer.train(train_iter_fct, args.train_steps)
    while len(neg_valid_loss) > 0:
        #from 3rd to 2nd to 1st.
        neg_loss, saved_model = heapq.heappop(neg_valid_loss)
        print(-neg_loss, saved_model)
        step = int(saved_model.split('.')[-2].split('_')[-1])
        test(args, device_id, saved_model, step)
    logger.info("Finish!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-bert_data_path", default='../bert_data/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')
    #parser.add_argument("-bert_config_path", default='../bert_config_uncased_base_small.json')
    parser.add_argument("-bert_config_path", default='config/bert_config_uncased_base.json')
    parser.add_argument("-model_name", default='ctx', type=str, choices=['base', 'ctx', 'seq'])
    parser.add_argument("-mode", default='train', type=str, choices=['train','validate','test','lead','oracle','getrouge'])
    parser.add_argument("-sent_sel_method", default='truth', type=str, choices=['truth','lead','model'])
    parser.add_argument("-max_label_sent_count", default=3, type=int)
    parser.add_argument("-label_format", default='soft', type=str, choices=['soft', 'greedy'], help = "soft:distribution from rouge scores; greedy: sentences with max 3-k(max sentetence limite - number of selected sentences) ROUGE scores set to 1 the other set to 0")
    parser.add_argument("-use_rouge_label", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-use_doc", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-salience_softmax", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-valid_by_rouge", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-loss", default='wsoftmax', type=str, choices=['bce','wsoftmax'])
    parser.add_argument("-aggr", default='last', type=str, choices=['last', 'mean_pool', 'max_pool'])
    # if it is last, only embeddng in step t-1 is used to predict setence in step t, same as in translation

    parser.add_argument("-rand_input_thre", default=1.0, type=float, help="the probability of keep the original labels of the selected sentences")
    parser.add_argument("-seg_count", default=30, type=int, help="how many segs to divide similarity score")
    parser.add_argument("-ngram_seg_count", default='20,20,20', type=str, help="seg count for unigram, bigram and trigram, has to be 3 int separated with comma" )
    parser.add_argument("-bilinear_out", default=10, type=int, help="dimension of the bilinear output")
    parser.add_argument("-temperature", default=20, type=float)
    parser.add_argument("-fix_scorer", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-max_epoch", default=2, type=int)
    parser.add_argument("-batch_size", default=3000, type=int)

    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=2048, type=int)
    parser.add_argument("-heads", default=8, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)
    parser.add_argument("-rnn_size", default=512, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=0.002, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='noam', type=str)
    parser.add_argument("-warmup_steps", default=10000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=2000, type=int)
    parser.add_argument("-save_model_count", default=3, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-world_size", default=1, type=int)
    parser.add_argument("-report_every", default=50, type=int)
    parser.add_argument("-train_steps", default=50000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='')
    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-report_precision", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    #this variable should not be set on gypsum
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.system('mkdir -p %s' % (os.path.dirname(args.log_file)))
    #if not args.bert_data_path.endswith("cnndm") and not args.bert_data_path.endswith("msword"):
    #    args.bert_data_path = args.bert_data_path + "/cnndm"
        #for notebook running, cnndm is not a directory name, it is prefix
    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if(args.world_size>1):
        multi_main(args)
    elif (args.mode == 'train'):
        train(args, device_id)
    elif (args.mode == 'validate'):
        wait_and_validate(args, device_id)
    elif (args.mode == 'lead'):
        baseline(args, cal_lead=True)
    elif (args.mode == 'oracle'):
        baseline(args, cal_oracle=True)
    elif (args.mode == 'test'):
        cp = args.test_from
        if cp == '':
            model_dir = args.model_path
            print(model_dir)
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            for cp in cp_files:
                step = int(cp.split('.')[-2].split('_')[-1])
                test(args, device_id, cp, step)
        else:
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test(args, device_id, cp, step)
    elif (args.mode == 'getrouge'):
        if args.model_name == 'base':
            pattern = '*step*initial.candidate'
        else:
            pattern = '*step*.candidate' #evaluate all
        candi_files = sorted(glob.glob("%s_%s" % (args.result_path, pattern)))
        #print(args.result_path)
        #print(candi_files)
        for can_path in candi_files:
            gold_path = can_path.replace('candidate', 'gold')
            rouge1_arr, rouge2_arr = du.compute_metrics(can_path, gold_path)
            step = os.path.basename(gold_path)
            precs_path = can_path.replace('candidate', 'precs')
            all_doc_ids = du.read_prec_file(precs_path)
            rouge_str, detail_rouge = test_rouge(args.temp_dir, can_path, gold_path, all_doc_ids, show_all=True)
            logger.info('Rouges at step %s \n%s' % (step, rouge_str))
            result_path = can_path.replace('candidate', 'rouge')
            if detail_rouge is not None:
                du.output_rouge_file(result_path, rouge1_arr, rouge2_arr, detail_rouge, all_doc_ids)
