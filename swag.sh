#!/usr/bin/env bash

python run_swag.py  \
       --data_dir=../SWAG  \
       --bert_model=../bert-large-cased-wwm/ \
       --output_dir=./outputs/swag/ \
       --do_train \
       --num_train_epochs=3  \
       --learning_rate=2e-5 \
       --train_batch_size=16 \
       --do_eval  