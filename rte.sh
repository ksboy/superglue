#!/usr/bin/env bash
# train_size =2490  sum_batch_size = 16*2 steps_per_epoch =79 epoch =8 sum_steps= 640

for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=2,3 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=../xlnet-large-cased-mnli/checkpoint-50000 \
    --do_train  \
    --do_eval   \
    --eval_all_checkpoints \
    --logging_steps=20 \
    --task_name=rte  \
    --data_dir=../data-superglue/RTE \
    --output_dir=./outputs/rte_xlnet/$SEED    \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate=1e-5 \
    --pop_layer=logits_proj \
    --save_steps=-1 \
    --seed=$SEED \
    --evaluate_during_training  \
    --num_train_epochs=8.0 \
    --warmup_steps=50 
    # --max_steps=1250 \
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --max_steps=1200  \
    # --model_name=xlnet-large-cased   \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \
    # --pop_classifier_layer  

done