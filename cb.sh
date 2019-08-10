#!/usr/bin/env bash
# train_size =250  sum_batch_size = 16*2 steps_per_epoch =8  epoches= 15 sum_steps = 120

BATCH_SIZE=16
# MAX_STEPS=120
EPOCHS=15.0
SAVE_STEPS=-1
LOG_STEPS=8
LR=1e-5

for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=0,2 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=../xlnet-large-cased-mnli/checkpoint-50000  \
    --do_train  \
    --do_eval   \
    --save_steps=$SAVE_STEPS \
    --eval_all_checkpoints \
    --logging_steps=$LOG_STEPS \
    --evaluate_during_training  \
    --task_name=cb  \
    --data_dir=../data-superglue/CB \
    --output_dir=./outputs/cb_xlnet/$SEED    \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate=$LR \
    --num_train_epochs=$EPOCHS \
    --warmup_steps=10 \
    --overwrite_output_dir  \
    --seed=$SEED 
    # --max_steps=120 \
    # --pop_layer=logits \
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --model_name=xlnet-large-cased   \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \

done