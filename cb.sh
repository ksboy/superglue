#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3 python run_classifier.py  \
       --data_dir=../data-superglue/CB  \
       --bert_model=../bert-large-cased-wwm-mnli/ \
       --task_name=cb \
       --output_dir=./outputs/cb10_remain_loss/ \
       --cache_dir=./cache/  \
       --do_train \
       --num_train_epochs=10 \
       --learning_rate=1e-5 \
       --train_batch_size=32 \
       --do_eval \
       --loss_weight=1,1,5
    #    --pop_classifier_layer  

