#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/COPA  \
       --bert_model=./outputs/copa16/ \
       --task_name=copa \
       --output_dir=./outputs/copa_predict/ \
       --cache_dir=./cache/  \
       --do_predict
    #    --pop_classifier_layer


