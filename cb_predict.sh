#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/CB  \
       --bert_model=./outputs/cb12/ \
       --task_name=cb \
       --output_dir=./outputs/cb_predict/ \
       --cache_dir=./cache/  \
       --do_predict
       #    --pop_classifier_layer \


