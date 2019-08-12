#!/usr/bin/env bash
# train_size =2490  sum_batch_size = 16*2 steps_per_epoch =79 epoch =10 sum_steps= 800
BATCH_SIZE=16
# MAX_STEPS=320
EPOCHS=10.0
SAVE_STEPS=-1
LOG_STEPS=40
LR=1e-5

for SEED in 3 7 42 50 87
do 
CUDA_VISIBLE_DEVICES=0,2 python run_superglue.py \
    --model_type=xlnet \
    --model_name_or_path=../xlnet-large-cased-mnli/checkpoint-50000 \
    --do_train  \
    --do_eval   \
    --eval_all_checkpoints \
    --logging_steps=$LOG_STEPS \
    --task_name=rte  \
    --data_dir=../data-superglue/RTE \
    --output_dir=./outputs/rte_xlnet/$SEED    \
    --cache_dir=./cache \
    --max_seq_length=128   \
    --per_gpu_eval_batch_size=$BATCH_SIZE   \
    --per_gpu_train_batch_size=$BATCH_SIZE   \
    --learning_rate=$LR \
    --pop_layer=logits_proj \
    --save_steps=$SAVE_STEPS \
    --seed=$SEED \
    --evaluate_during_training  \
    --num_train_epochs=$EPOCHS \
    --warmup_steps=50 \
    --overwrite_output_dir   

    # --max_steps=1250 \
    # --tokenizer_name=xlnet_large_cased  \
    # --gradient_accumulation_steps=1 \
    # --max_steps=1200  \
    # --model_name=xlnet-large-cased   \
    # --do_lower_case \
    # --overwrite_cache \
    # --pop_classifier_layer  

done