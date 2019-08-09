#!/usr/bin/env bash
# train_size =400  sum_batch_size = 32 steps_per_epoch =13  epoch =10 sum_steps= 130

BATCH_SIZE=32
MAX_STEPS=130
SAVE_STEPS=-1
LOG_STEPS=10
LR=1e-5

for SEED in 42
do 
CUDA_VISIBLE_DEVICES=2 python run_superglue.py \
    --model_type=bert \
    --model_name_or_path=./outputs/bert-large-cased-swag/ \
    --do_train  \
    --do_eval   \
    --eval_all_checkpoints \
    --logging_steps=$LOG_STEPS \
    --task_name=copa  \
    --data_dir=../data-superglue/COPA \
    --output_dir=./outputs/copa_bert_v2/$BATCH_SIZE/$SEED   \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate=$LR \
    --pop_layer=classifier\
    --max_steps=$MAX_STEPS \
    --save_steps=$SAVE_STEPS \
    --evaluate_during_training  \
    --num_train_epochs=10.0 \
    --seed=$SEED 
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --max_steps=1200  \
    # --model_name=xlnet-large-cased   \
    # --warmup_steps=120 \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \
    # --pop_classifier_layer  

done