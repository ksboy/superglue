#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/COPA  \
       --bert_model=./outputs/swag/ \
       --pop_classifier_layer \
       --task_name=copa \
       --output_dir=./outputs/copa24/ \
       --cache_dir=./cache/  \
       --do_train \
       --num_train_epochs=8  \
       --learning_rate=1e-5 \
       --train_batch_size=16 \
       --do_eval  