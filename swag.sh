#!/usr/bin/env bash
# train_size =73545  sum_batch_size = 16 steps_per_epoch =4597 sum_steps= 13791


CUDA_VISIBLE_DEVICES=1,3 python run_swag.py \
    --model_type=xlnet \
    --model_name_or_path=../xlnet-large-cased \
    --do_train  \
    --do_eval   \
    --save_steps=3000 \
    --eval_all_checkpoints \
    --logging_steps=300 \
    --evaluate_during_training  \
    --task_name=swag  \
    --data_dir=../SWAG \
    --output_dir=./outputs/xlnet-large-cased-swag   \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=4 \
    --per_gpu_train_batch_size=4 \
    --learning_rate=1e-5 \
    --num_train_epochs=3.0
    # --max_steps= 
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --max_steps=1200  \
    # --model_name=xlnet-large-cased   \
    # --warmup_steps=120 \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \
    # --pop_classifier_layer  

