#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3 python run_classifier.py  \
       --data_dir=../data-superglue/CB  \
       --bert_model=../bert-large-cased-wwm-mnli/ \
       --task_name=cb \
       --output_dir=./outputs/cb15_remain/ \
       --cache_dir=./cache/  \
       --do_train \
       --num_train_epochs=12 \
       --learning_rate=1e-5 \
       --train_batch_size=16 \
       --do_eval \
    #    --pop_classifier_layer  

