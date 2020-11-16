# ARedSumSentRank
Code for Paper ["Adaptive Redundancy-Aware Iterative Sentence Ranking for Extractive Document Summarization"](https://arxiv.org/abs/2004.06176).
This code (ARedSumSentRank) has two versions of adaptive redundancy-aware sentence ranking models: 
**ARedSumCTX and ARedSumSEQ**

This code is based on [BertSum](https://github.com/nlpyang/BertSum), pytorch 1.1.0, and python 3.6.9.
In order to run this code, you need the following packages.
```
    pip install multiprocess
    pip install tensorboardX
    pip install pyrouge (first install rouge)
    pip install pytorch_pretrained_bert
```
## Preprocessing Stage 1:
### CNN/DailyMail
The preprocessing steps for CNN/DailyMail follows [BertSum](https://github.com/nlpyang/BertSum) using the non-anonymized version.
#### Option 1: Download the [preprocessed data](https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6)
#### Option 2: Preprocess the data yourself.
1. First, download [stories](http://cs.nyu.edu/~kcho/DMQA/) and put all the story files into raw_stories.
  
2. Download the [StandfordNLP package](https://stanfordnlp.github.io/CoreNLP/), unzip it and add this line to your bash profile:
  ```
    export CLASSPATH=/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar 
  ```
Replace `/path/to/` with the path where the package is actually saved.
    
3. Download the `urls` directory from [BertSum](https://github.com/nlpyang/BertSum) and put it in directory `ARedSumSentRank`. These files are important for dividing data.
### New York Times
1. Get New York Times [T19_data](https://catalog.ldc.upenn.edu/LDC2008T19)

2. Filter and extract the original data to get stories for NYT50:
   ```
   python process_nyt_t19.py -input_path /path/to/T19_data -output_path /path/to/working/dir
   ```
    
## Preprocessing Stage 2:
Afterwards the preprocessing steps for both datasets are as follows:
Go to ARedSumSentRank/src

*Step1: Tokenize the stories with StanfordNLP*
```
  python preprocess.py -mode tokenize -raw_path /path/to/raw_stories -save_path /path/to/tokenized/stories
```
*Step2: Extract source and target sentences from the tokenized data save train, validation and test according to the given urls.*
```
  python preprocess.py -mode format_to_lines -raw_path /path/to/cnndm_tokenized_stories -save_path /path/to/cnndm/json_data/ -map_path ../urls/ -lower -n_cpus 4
```
*Step3: Convert the data to Bert format.*
  (filtering, cut long sentence, get labels based on rouge)
```
  python preprocess.py -mode format_to_bert -raw_path /path/to/cnndm/json_data/cnndm -save_path /path/to/cnndm/bert_data/ -oracle_mode greedy -n_cpus 4 -log_file ../logs/preprocess.log
```
   Note: in each step, `-raw_path` requires the input path, `-save_path` requires the path to save the output.
    
   For NYT50, use `nyt_preprocess.py` instead of `preprocess.py`, and change the path name to the corresponding path of NYT50.

## Parameters:
-mode:

    "train":
        Train model and validate the model every a few thousand steps, keep the models with top args.save_model_count loss and delete the others to save some space. At last, the final args.save_model_count models will be tested on the test collection. (predicted summaries will be saved) Self-computed Rouge will be output in the log. Perl Rouge will also be output.
        If you want to disable the calculation of the Perl Rouge, add
            cmd_arr.append("-report_rouge False")
        in the code.
    "test":
        When there are models in the directory, test the model with test set and output the predicted summaries. Output Rouge scores.
    "getrouge":
        Compute perl version rouge with the predicted summaries. Usually called after the predicted summaries have already been predicted.
-model_name:

    "base": BertSum + bilinear matching + listwise loss
    "ctx": ARedSumCtx model in the paper
    "seq": ARedSumSeq model in the paper

-max_epoch:

    Max number of epochs to train
-train_steps:

    How many steps to train; whichever stops the training first (max_epoch, train_steps)
-rouge_label:

    Use rouge-based label or not
    "t" or "f", "True" or "False", "1" or "0" to indicate
-valid_by_rouge:

    Use rouge score to do validation and select best args.save_model_count models, otherwise use loss to choose models.
-loss:

    "bce": Binary cross entropy loss
    "wsoftmax": Weighted softmax (no normalization to label, only softmax normalization to predicted scores)
    when rouge based labels are used, loss is always cross entropy between softmax of rouge labels and predicted scores.
-use_doc:

    Use document embeddings as context or not. currently only applies to seq2seq model.
-rand_input_thre:

    The probability that the selected sentences are kept as original.
    With 1-rand_input_thre probability, selected sentences will be replaced with random sentences in the document.

## A Simple Way to Train Models
To run different settings of training and evaluating models, go to src/ run:
  ```
  python run_para.py
  ```
The hypter-parameters have been set in the script. You need to set the `DATA_ROOT_DIR`, `DATA_PATH`, `MODEL_DIR` and `RESULT_DIR` to a valid place. The hyper-parameters and the settings in `run_para.py` are the settings corresponding to the paper. The results from re-running models may be a little different from what are reported in the paper. Models and predicted summaries correponding to the reported numbers in the paper can be provided upon request. The models are fine-tuned for `train_steps` and `max_epoch` whichever is reached first. Overall, 50000 steps are used as the stop condition in our models.

By default, during training, after each `-save_checkpoint_steps` the model will be validated on the validation set and at last only models with the top 3 performances (in terms of either loss or ROUGE scores indicated by `-valid_by_rouge`) will be saved and tested with the test data. Precision and ROUGE scores will be reported by default. Whether to calculate precision and ROUGE can be controlled with `-report_precision` and `-report_rouge` respectively.

Note that the multi-GPU training has not been tested. Be carefull if you use the multi-GPU setting.
The individual commands to train each model are as follows:
### (ARedSum-CTX) Train a Salience Ranker
```
python train.py -bert_data_path /path/to/cnndm_or_nyt50/bert_data/ -visible_gpus 0 -gpu_ranks 0 -accum_count 2 -report_every 50 -save_checkpoint_steps 2000 -decay_method noam -mode train -model_name base -label_format soft -result_path /path/to/results/cnndm_or_nyt50 -model_path /path/to/model/cnn_or_nyt50
```
Set `-model_path` to the place where you want to save the model and set `-result_path` to the place where the predicted summaries are stored
### (ARedSum-CTX) Train a Ranker for Selection
```
python train.py -fix_scorer -train_from /path/to/the/best/salience/ranker.pt -bert_data_path /path/to/cnndm_or_nyt50/bert_data/ -visible_gpus 2 -gpu_ranks 0 -accum_count 2 -report_every 50 -save_checkpoint_steps 2000 -decay_method noam -model_name ctx -max_epoch 2 -train_steps 50000 -label_format soft -use_rouge_label t -valid_by_rouge t -rand_input_thre 1.0 -temperature 20 -seg_count 30 -ngram_seg_count 20,20,20 -bilinear_out 20 -result_path /path/to/where/you/want/to/save/the/preidicted/summaries -model_path /path/to/where/you/want/to/save/the/models
```
For NYT50, change the settings in previous command as follows:
```
-rand_input_thre 0.8 -temperature 20 -seg_count 30 -ngram_seg_count 10,10,10 -bilinear_out 20
```
To get the performances without Trigram Blocking, set `-block_trigram False`. By default, it is set to True.
### (ARedSum-SEQ) Train a Sequence Generation Model
```
python train.py -bert_data_path /path/to/cnndm_or_nyt50/bert_data/ -visible_gpus 2 -gpu_ranks 0 -accum_count 2 -report_every 50 -save_checkpoint_steps 2000 -decay_method noam -model_name seq -max_epoch 2 -train_steps 50000 -label_format soft -use_rouge_label t -valid_by_rouge t -rand_input_thre 0.8 -temperature 20 -result_path /path/to/where/you/want/to/save/the/preidicted/summaries -model_path /path/to/where/you/want/to/save/the/models
```

## Validate or Test the Model
Simple change`-mode train` to `-mode validate` or `-mode test`. When `-mode test` is used, a specific model name can be fed to `-test_from`, otherwise, all the models in the directory input to `-model_path` will be applied on the test data.
`-mode get_rouge` can be used to evaluate the ROUGE scores of the predicted summaries in `-result_path`.

## Run Baselines
Get baseline (Lead 3 or Oracle) performances:
```
python train.py -mode lead/oracle -bert_data_path /path/to/cnndm_or_nyt50/bert_data/ -result_path /path/to/where/you/want/to/save/the/preidicted/summaries -block_trigram False
```
`-block_trigram True` can be applied too if you want to get the performances of lead/oracle + Trigram Blocking

