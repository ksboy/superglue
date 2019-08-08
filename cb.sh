
#!/usr/bin/env bash
# train_size =250  sum_batch_size = 32 steps_per_epoch =8 epoch =12 sum_steps= 100
BATCH_SIZE=32
MAX_STEPS=100
SAVE_STEPS=-1
LOG_STEPS=8
LR=1e-5

for SEED in 3
do 
CUDA_VISIBLE_DEVICES=2 python run_superglue.py \
    --model_type=bert \
    --model_name_or_path=../bert-large-cased-wwm-mnli/ \
    --do_train  \
    --do_eval   \
    --eval_all_checkpoints \
    --logging_steps=$LOG_STEPS \
    --task_name=cb  \
    --data_dir=../data-superglue/CB \
    --output_dir=./outputs/cb_bert/$BATCH_SIZE/$SEED   \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate=$LR \
    --max_steps=$MAX_STEPS \
    --save_steps=$SAVE_STEPS \
    --seed=$SEED \
    --evaluate_during_training  \
    --num_train_epochs=12.0 
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --max_steps=1200  \
    # --model_name=xlnet-large-cased   \
    # --warmup_steps=120 \
    # --do_lower_case \
    # --overwrite_output_dir   \
    # --overwrite_cache \
    # --pop_layer=classifier\

done