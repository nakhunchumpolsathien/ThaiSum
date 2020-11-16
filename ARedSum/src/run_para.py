import os
import sys

DATA_ROOT_DIR="/net/home/kbi/ingham_disk/doc_summarization"

DATA_PATH="%s/CNNDM/bert_data/cnndm" % DATA_ROOT_DIR
MODEL_DIR="%s/models/cnndm/" % DATA_ROOT_DIR
LOG_DIR= "/net/home/kbi/projects/doc_summarization/ARedSumSentRank/logs"
RESULT_DIR="%s/outputs" % DATA_ROOT_DIR

#script_path = "sudo /data/anaconda/envs/py36/bin/python train.py"
script_path = "python train.py"
CONST_CMD_ARR = [("bert_data_path", DATA_PATH),
        ("visible_gpus", "2"),
        ("gpu_ranks", "0"),
        ("accum_count", 2),
        ("report_every", 50),
        ("save_checkpoint_steps", 2000),
        ("decay_method", "noam")]
CONST_CMD = " ".join(["-{} {}".format(x[0], x[1]) for x in CONST_CMD_ARR])
EVAL_CMD = "-test_all"

para_names = ['mode', 'model_name', 'max_epoch', 'train_steps', 'label_format', 'use_rouge_label', 'valid_by_rouge', \
        'use_doc', 'rand_input_thre', 'temperature', \
        'seg_count', 'ngram_seg_count', 'bilinear_out']
short_names = ['', 'mn', 'me', 'ts', 'if', 'rl', 'vbr', 'ud', 'rit', 'tprt',\
                'sc', 'nsc', 'bo']
paras = [
        ('train', 'base', 2, 50000, 'soft', 'f','f', False, 1.0, 0, 1, '1,1,1',1),
        ('train', 'ctx', 2, 50000, 'soft', 't','t', False, 1.0, 20, 30, '20,20,20',20),
        ('train', 'seq', 2, 50000, 'soft', 't','t', True, 0.8, 20, 1, '1,1,1',1),
        ]

nyt_paras = [
        ('train', 'base', 2, 50000, 'soft', 'f','f', False, 1.0, 0, 1, '1,1,1',1),
        ('train', 'ctx', 2, 50000, 'soft', 't','t', False, 0.8, 20, 30, '10,10,10',20),
        ('train', 'seq', 2, 50000, 'soft', 't','t', True, 0.8, 20, 1, '1,1,1',1),
        ]


for para in paras:
    cmd_arr = []
    cmd_arr.append(script_path)
    model_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names, para)][1:])
    #train or valid or test not need to be included
    result_path = "%s/%s/cnndm" % (RESULT_DIR, model_name)

    model_path = "%s/%s" % (MODEL_DIR, model_name)
    cur_cmd_option = " ".join(["-{} {}".format(x,y) for x,y in zip(para_names, para)])
    mode = para[0]
    if mode == "train":
        batch_size = 3000
    elif mode == "validate":
        batch_size = 30000
        cmd_arr.append(EVAL_CMD)
    else:
        batch_size = 30000
    #cmd_arr.append("-report_rouge False")
    cmd_arr.append("-batch_size %s" % batch_size)

    if para[1] == 'ctx':
        cmd_arr.append("-fix_scorer")
        saved_model_name ="/mnt/scratch/kbi/doc_summarization/models/cnndm/ectransformer_group_me2_ts50000_ifgroup_gs1_labelremove_te2_rlf_vbrf_losswsoftmax_scorerbilinear_oiTrue_ssfirst_udTrue_rit1.0_tprt0/model_step_48000.pt"
        cmd_arr.append("-train_from %s" % saved_model_name)

    cmd_arr.append(CONST_CMD)
    cmd_arr.append(cur_cmd_option)
    cmd_arr.append("-result_path %s" % result_path)
    cmd_arr.append("-model_path %s" % model_path)
    cmd_arr.append("-log_file %s/%s_%s.log" % (LOG_DIR, mode, model_name))
    cmd_arr.append("&> %s_%s.log" % (mode, model_name))
    cmd = " " .join(cmd_arr)
    print(cmd)
    os.system(cmd)

