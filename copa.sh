#!/usr/bin/env bash
# train_size =400  sum_batch_size = 16*2 steps_per_epoch =13  epoch =12 sum_steps= 156

BATCH_SIZE=16
# MAX_STEPS=300
EPOCHS=12.0
SAVE_STEPS=-1
LOG_STEPS=13
LR=1e-5

for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=2,3 python run_copa.py \
    --model_type=bert \
    --model_name_or_path=./outputs/bert-large-cased-swag/ \
    --do_train  \
    --do_eval   \
    --eval_all_checkpoints \
    --logging_steps=$LOG_STEPS \
    --task_name=copa  \
    --data_dir=../data-superglue/COPA \
    --output_dir=./outputs/copa_bert/$SEED   \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate=$LR \
    --pop_layer=classifier\
    --save_steps=$SAVE_STEPS \
    --evaluate_during_training  \
    --num_train_epochs=$EPOCHS \
    --seed=$SEED \
    --overwrite_output_dir  
    # --max_steps=$MAX_STEPS \
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