#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/RTE  \
       --bert_model=./outputs/rte2/ \
       --task_name=rte \
       --output_dir=./outputs/rte_eval_tmp/ \
       --cache_dir=./cache/  \
       --do_eval
    #    --pop_classifier_layer