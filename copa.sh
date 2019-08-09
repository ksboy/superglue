#!/usr/bin/env bash
# train_size =400*2  sum_batch_size = 16*2 steps_per_epoch =25  sum_steps= 250

for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=0,3 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=../superglue/outputs/xlnet-large-cased-swag/checkpoint-27000 \
    --do_train  \
    --do_eval   \
    --save_steps=-1 \
    --eval_all_checkpoints \
    --logging_steps=10 \
    --evaluate_during_training  \
    --task_name=copa  \
    --data_dir=../data-superglue/COPA \
    --output_dir=./outputs/copa_xlnet/$SEED   \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate=1e-5 \
    --num_train_epochs=10.0 \
    --max_steps=250 \
    --warmup_steps=15 \
    --pop_layer=logits \
    --seed=$SEED \
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \

done