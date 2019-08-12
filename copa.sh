#!/usr/bin/env bash
# train_size =400  sum_batch_size = 8*2 steps_per_epoch =25  sum_steps= 300

BATCH_SIZE=8
# MAX_STEPS=300
EPOCHS=12.0
SAVE_STEPS=-1
LOG_STEPS=20
LR=1e-5

for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=2,3 python run_copa.py \
    --model_type=xlnet \
    --model_name_or_path=../superglue/outputs/xlnet-large-cased-swag/checkpoint-27000 \
    --do_train  \
    --do_eval   \
    --eval_all_checkpoints \
    --logging_steps=$LOG_STEPS \
    --evaluate_during_training  \
    --task_name=copa  \
    --data_dir=../data-superglue/COPA \
    --output_dir=./outputs/copa_xlnet/$SEED   \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate=$LR \
    --num_train_epochs=$EPOCHS \
    --warmup_steps=15 \
    --pop_layer=logits \
    --save_steps=$SAVE_STEPS \
    --seed=$SEED \
    --overwrite_output_dir  

    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \

done