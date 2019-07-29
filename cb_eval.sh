#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2,3 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=./outputs/cb20/ \
    --do_eval   \
    --task_name=cb  \
    --data_dir=../data-superglue/CB \
    --output_dir=./proc_data/cb20   \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=8   \
    # --gradient_accumulation_steps=1 \
    # --max_steps=1200  \
    # --model_name=xlnet-large-cased   \
    # --warmup_steps=120 \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \
    # --pop_classifier_layer  

