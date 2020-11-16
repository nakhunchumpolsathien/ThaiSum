import os
import re
import shutil
import time
import pandas as pd

from others import pyrouge

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)
    #group() by default group(0), i.e., the entire match
    #replace the first argument in x by the second argument


def process(params):
    temp_dir, data = params
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        #print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict

def test_rouge(temp_dir, cand, ref, doc_ids=None, show_all=True):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])

            multi_refs = references[i].split("<JG>")
            for ref_idx in range(len(multi_refs)):
                with open(tmp_dir + "/reference/ref.{}.{}.txt".format(ref_idx, i), "w",
                          encoding="utf-8") as f:
                    f.write(multi_refs[ref_idx])

        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.\d+.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        #print(rouge_results)
        if show_all:
            df = r.output_to_dataframe(rouge_results)
            df.index = doc_ids + ["average"]
            #df = pd.concat([df[:-1].sort_index(), df[-1:]], axis=0)
            output = df[-1:].to_dict("records")[0]
            display_metrics = ["RG%s-%s" % (n, m) for n in ['1','2','L'] for m in ['P', 'R', 'F']]
            metrics = ["rouge-%s-%s" % (n, m) for n in ['1','2','L'] for m in ['P', 'R', 'F']]
            #metrics_head = "\t".join(metrics)
            metrics_numbers = ["%.2f" % (output[x] * 100) for x in metrics]
            #metrics_num_str = "\t".join(metrics_numbers)
            #result_str = "%s\n%s\n" % (metrics_head, metrics_num_str)
            result_str = "\t".join(["%s:%s" % (x,y) for x,y in zip(display_metrics, metrics_numbers)])

            records = df[:-1].to_dict("records")
            detail_results = {"individual": {doc_id: record
                #for doc_id, record in zip(doc_ids, records)},
                #this way the order of doc_ids and the corresponding rouge is messed up.
                for doc_id, record in zip(df.index[:-1], records)},
                       "average": df[-1:].to_dict("records")[0]}
        else:
            results_dict = r.overall_output_to_dict(rouge_results)
            result_str = rouge_results_to_str(results_dict)
            detail_results = None
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
            #pass
    return result_str, detail_results


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )
