#!/usr/bin/env bash
# train_size =2490  sum_batch_size = 16 steps_per_epoch =156 sum_steps= 2340


CUDA_VISIBLE_DEVICES=0,3 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=./outputs/cb25/checkpoint-240/ \
    --do_predict   \
    --task_name=cb  \
    --data_dir=../data-superglue/CB \
    --output_dir=./outputs/cb25/checkpoint-240/  \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8  
    # --do_predict
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

