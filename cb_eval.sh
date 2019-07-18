#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/CB  \
       --bert_model=./outputs/cb4/ \
       --pop_classifier_layer=false \
       --task_name=cb \
       --output_dir=./outputs/cb7/ \
       --cache_dir=./cache/  \
       --do_eval 


