#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/RTE  \
       --bert_model=./outputs/mnli2/ \
       --task_name=rte \
       --output_dir=./outputs/rte11_pop/ \
       --cache_dir=./cache/  \
       --do_train \
       --num_train_epochs=5  \
       --learning_rate=1e-5 \
       --train_batch_size=32 \
       --do_eval \
       --pop_classifier_layer 

