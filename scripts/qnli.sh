#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../glue_data/QNLI  \
       --bert_model=../bert-large-cased-wwm/ \
       --task_name=qnli \
       --output_dir=./outputs/qnli/ \
       --cache_dir=./cache/  \
       --do_train \
       --num_train_epochs=3  \
       --learning_rate=1e-5 \
       --train_batch_size=32 \
       --do_eval  
