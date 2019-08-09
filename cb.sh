#!/usr/bin/env bash
# train_size =250  sum_batch_size = 16*2 steps_per_epoch =8  epoches= 15 sum_steps = 120

for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=0,3 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=../xlnet-large-cased-mnli/checkpoint-50000  \
    --do_train  \
    --do_eval   \
    --save_steps=-1 \
    --eval_all_checkpoints \
    --logging_steps=8 \
    --evaluate_during_training  \
    --task_name=cb  \
    --data_dir=../data-superglue/CB \
    --output_dir=./outputs/cb_xlnet_remain/$SEED    \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate=1e-5 \
    --num_train_epochs=15.0 \
    --max_steps=120 \
    --warmup_steps=10 \
    --seed=$SEED 
    # --pop_layer=logits \
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --model_name=xlnet-large-cased   \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \

done