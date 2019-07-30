#!/usr/bin/env bash
# train_size =2490  sum_batch_size = 16 steps_per_epoch =156 sum_steps= 2340


CUDA_VISIBLE_DEVICES=0,3 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=./outputs/rte13/checkpoint-400/ \
    --do_eval   \
    --task_name=rte  \
    --data_dir=../data-superglue/RTE \
    --output_dir=./outputs/rte13/checkpoint-400/  \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8  
    # --tokenizer_name=xlnet_large_cased  
    # --evaluate_during_training  \
    # --num_train_epochs=15.0 \
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --max_steps=1200  \
    # --model_name=xlnet-large-cased   \
    # --warmup_steps=120 \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \
    # --pop_classifier_layer  

