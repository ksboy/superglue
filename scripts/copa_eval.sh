#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/COPA  \
       --bert_model=./outputs/swag/ \
       --task_name=copa \
       --output_dir=./outputs/eval/ \
       --cache_dir=./cache/  \
       --do_eval  
    #    --pop_classifier_layer