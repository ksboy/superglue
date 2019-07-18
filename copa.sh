#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/COPA  \
       --bert_model=../bert-large-cased-wwm-mnli/ \
       --pop_classifier_layer=true \
       --task_name=copa \
       --output_dir=./outputs/copa18/ \
       --cache_dir=./cache/  \
       --do_train \
       --num_train_epochs=6  \
       --learning_rate=1e-5 \
       --train_batch_size=32 \
       --do_eval  