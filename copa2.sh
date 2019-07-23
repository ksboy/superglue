#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,2,3 python run_copa.py  \
       --data_dir=../data-superglue/COPA  \
       --bert_model=./outputs/swag/ \
       --output_dir=./outputs/copa_temp2/ \
       --do_train \
       --num_train_epochs=6  \
       --learning_rate=1e-5 \
       --train_batch_size=32 \
       --do_eval  \
    #    --pop_classifier_layer 