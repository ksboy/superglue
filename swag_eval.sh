#!/usr/bin/env bash

python run_swag.py  \
       --data_dir=../SWAG  \
       --bert_model=./outputs/swag \
       --output_dir=./outputs/swag_eval/ \
       --eval_batch_size=128 \
       --do_eval  