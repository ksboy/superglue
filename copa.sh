#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/COPA  \
       --bert_model=../bert-base-cased-finetuned-mrpc/ \
       --task_name=copa \
       --output_dir=./copa/ \
       --cache_dir=./cache/  \
       --do_train \
       --do_eval
