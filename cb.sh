#!/usr/bin/env bash

python run_classifier.py  \
       --data_dir=../data-superglue/CB  \
       --bert_model=../bert-large-cased-wwm-mnli/ \
       --pop_classifier_layer=false \
       --task_name=cb \
       --output_dir=./outputs/cb14/ \
       --cache_dir=./cache/  \
       --do_train \
       --num_train_epochs=12 \
       --learning_rate=1e-5 \
       --train_batch_size=32 \
       --do_eval 

