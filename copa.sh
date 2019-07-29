#!/usr/bin/env bash
# train_size =400  sum_batch_size = 16 steps_per_epoch =25  sum_steps= 480


CUDA_VISIBLE_DEVICES=0,2,3 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=./outputs/xlnet-large-cased-swag/checkpoint-27000 \
    --do_train  \
    --do_eval   \
    --save_steps=80 \
    --eval_all_checkpoints \
    --logging_steps=10 \
    --evaluate_during_training  \
    --task_name=cb  \
    --data_dir=../data-superglue/COPA \
    --output_dir=./outputs/copa25   \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate=1e-5 \
    --num_train_epochs=30.0
    # --max_steps= 
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --max_steps=1200  \
    # --model_name=xlnet-large-cased   \
    # --warmup_steps=120 \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \
    --pop_classifier_layer  

