#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/CB  \
       --bert_model=../bert-base-cased-finetuned-mrpc/ \
       --task_name=cb \
       --output_dir=./cb/ \
       --cache_dir=./cache/  \
       --do_train \
       --do_eval
