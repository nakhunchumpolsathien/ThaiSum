import gc
import os
import glob
import argparse
import subprocess
import json
from os.path import join as pjoin
from others.utils import clean
from others.logging import init_logger
from others.logging import logger
import torch
from multiprocess import Pool
import pathlib
from pytorch_pretrained_bert import BertTokenizer
from datetime import datetime
from prepro.data_builder import BertData, greedy_selection

#1. Convert train.tsv, dev.tsv, test.tsv to files that contain only one sentence each.
#   these files are used for tokenization only with StanfordNLP tokenizer.
#2. In the same time, get the labels for each document at paragraph, section, doc and file card level.
#   Only merged judgements from judges can be obtained from train.tsv, dev.tsv and test.tsv.
#3. Run StanfordNLP tokenizer to tokenize each sentence and output to json files.
#4. Merge json files with tokenized sentences to src (list of senteneces which consist list of tokens), tgt style, each line represents one doc.
#5. Load the files from Step 4, and convert it to bert format

def tokenize(args):
    sent_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)
    print("Preparing to tokenize %s to %s..." % (sent_dir, tokenized_stories_dir))
    sentences = os.listdir(sent_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    num_orig = 0
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in sentences:
            if (not s.endswith('story')):
                continue
            num_orig += 1
            f.write("%s\n" % (os.path.join(sent_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP' ,'-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(sentences), sent_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized sentences directory contains the same number of files as the original directory
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized sentences directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, sent_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (sent_dir, tokenized_stories_dir))

def format_to_lines(args, split_ratio=[0.8,0.1,0.1]):
    output_dir = os.path.dirname(args.save_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    file_list = os.listdir(args.raw_path)
    file_list.sort(key = lambda f: (datetime.strptime(f.rsplit("_", 1)[0], '%Y_%m_%d'), int(f.rsplit("_", 1)[1].split(".")[0])))
    file_list = ["%s/%s" % (args.raw_path, f) for f in file_list]
    #print(file_list)
    train_count, valid_count, test_count = [round(len(file_list) * x) for x in split_ratio]
    print(train_count, valid_count, test_count)

    train_files = file_list[:train_count]
    valid_files = file_list[train_count:train_count+valid_count]
    test_files = file_list[train_count+valid_count:]

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            #randomly assigned the entries in a_lst to different processors in the pool
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []

def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            continue
        if (flag):
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt


def _format_to_lines(params):
    f, args = params
    #print(f)
    doc_id = os.path.basename(f).split('.')[-3]
    print(doc_id)
    source, tgt = load_json(f, args.lower)
    return {'docId': doc_id, 'src': source, 'tgt': tgt}

def format_to_bert(args):
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        #print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass

        pool.close()
        pool.join()

def _format_to_bert(params, sent_count=5):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        doc_id, source, tgt = d['docId'], d['src'], d['tgt']
        if (args.oracle_mode == 'greedy'):
            oracle_ids = greedy_selection(source, tgt, sent_count)
        elif (args.oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, sent_count)
        #print(oracle_ids)
        b_data = bert.preprocess(source, tgt, oracle_ids)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        #print(labels)
        b_data_dict = {"doc_id":doc_id, "src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

def _format_to_nnsum(params, sent_count=5):
    f, args, input_dir, abstracts_dir, label_dir = params
    #print(f)
    doc_id = f.split('/')[-1].split('.')[0] #0000bf554ca24b0c72178403b54c0cca62d9faf8.story.json
    source, tgt = load_json(f, args.lower)
    if len(source) < 1 or len(tgt) < 1:
        return
    if (args.oracle_mode == 'greedy'):
        oracle_ids = greedy_selection(source, tgt, sent_count)
    elif (args.oracle_mode == 'combination'):
        oracle_ids = combination_selection(source, tgt, sent_count)

    '''we should filter the empty file here'''
    labels = [1 if idx in oracle_ids else 0 for idx in range(len(source))]
    label_str = {"id":doc_id, "labels": labels}
    label_file = label_dir / "{}.json".format(doc_id)
    label_file.write_text(json.dumps(label_str))

    inputs = [{"tokens": sent, "text": " ".join(sent)} for sent in source]
    entry = {"id":doc_id, "inputs":inputs}
    input_file = input_dir / "{}.json".format(doc_id)
    input_file.write_text(json.dumps(entry))

    lines = [" ".join(sent) for sent in tgt]
    target_str = "\n".join(lines)
    abstract_file = abstracts_dir / "{}.spl".format(doc_id)
    abstract_file.write_text(target_str)
    '''
    '''


def format_to_nnsum(args, split_ratio=[0.8,0.1,0.1]):
    ''' convert data to what nnsum(https://github.com/kedz/nnsum) can use
        for training SummaRunner and other baseline models.
    label_file: {id}.json
            {"id":"7f168bcf16ff08b32221d0c3993701dd502de584",
            "labels":[1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
    abstract_file: {id}.spl
            # nnsum paper uses tokenized words joined by space as each sentence,
            but uncased (both upper and lower case included)
    input_file: {id}.json
            {"input": [sent_1, sent_2, ..., sent_n], "id":story_id}
            sent_i: {"text":original text, "tokens":word list, "pos":postag, "ne":NER,
                    "word_count":word count of sent_i, "sentence_id":i}
            #sentence_id is from 1
            #The fields really used in the model are:
                "tokens", "text"
    '''
    output_dir = os.path.dirname(args.save_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    file_list = os.listdir(args.raw_path)
    file_list.sort(key = lambda f: (datetime.strptime(f.rsplit("_", 1)[0], '%Y_%m_%d'), int(f.rsplit("_", 1)[1].split(".")[0])))
    file_list = ["%s/%s" % (args.raw_path, f) for f in file_list]
    #print(file_list)
    train_count, valid_count, test_count = [round(len(file_list) * x) for x in split_ratio]
    print(train_count, valid_count, test_count)

    train_files = file_list[:train_count]
    valid_files = file_list[train_count:train_count+valid_count]
    test_files = file_list[train_count+valid_count:]

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        data_dir = pathlib.Path(args.save_path)
        input_dir = data_dir / "nnsum_inputs" / corpus_type
        label_dir = data_dir / "nnsum_labels" / corpus_type
        abstracts_dir = data_dir / "human-abstracts" / corpus_type
        input_dir.mkdir(exist_ok=True, parents=True) # similar to 'mkdir -p'
        label_dir.mkdir(exist_ok=True, parents=True)
        abstracts_dir.mkdir(exist_ok=True, parents=True)
        a_lst = [(f, args, input_dir, abstracts_dir, label_dir) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        result_iter = pool.imap_unordered(_format_to_nnsum, a_lst)

        num_stories = len(a_lst)
        #randomly assigned the entries in a_lst to different processors in the pool
        for idx, result in enumerate(result_iter, 1):
            print(
                "{}: Writing story {}/{}".format(corpus_type, idx, num_stories),
                end="\r" if idx < num_stories else "\n",
                flush=True)

        pool.close()
        pool.join()
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str, help='format_to_lines or format_to_bert')
    parser.add_argument("-raw_path", default='/mnt/scratch/kbi/doc_summarization/NYT/nyt_stories')
    parser.add_argument("-save_path", default='/mnt/scratch/kbi/doc_summarization/NYT/nyt_tokenized_stories')
    parser.add_argument("-oracle_mode", default='greedy', type=str, help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')

    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)

    parser.add_argument('-log_file', default='../logs/nyt_preprocess.log')
    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')
    parser.add_argument('-n_cpus', default=4, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    if args.mode == 'tokenize':
        tokenize(args)
    elif args.mode == 'format_to_lines':
        format_to_lines(args)
    elif args.mode == 'format_to_bert':
        format_to_bert(args)
    elif args.mode == 'format_to_nnsum':
        format_to_nnsum(args)
