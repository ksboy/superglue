#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/RTE  \
       --bert_model=../bert-base-cased-finetuned-mrpc/ \
       --task_name=rte \
       --output_dir=./rte/ \
       --cache_dir=./cache/  \
       --do_train \
       --do_eval
