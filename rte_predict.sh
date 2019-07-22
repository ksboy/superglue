#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/RTE  \
       --bert_model=./outputs/rte7/ \
       --task_name=rte \
       --output_dir=./outputs/rte_predict7/ \
       --cache_dir=./cache/  \
       --do_predict
    #    --pop_classifier_layer


