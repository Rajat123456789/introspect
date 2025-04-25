#!/bin/bash
            
data_path="data/finetune_data.json"
output_path="introspect-llama-3.3-7b-instruct"
output_path="introspect-llama-3.3-13b-instruct"

torchrun --nproc_per_node=4 --master_port=2023 ../introspect-medalpaca/train.py \
    --model "introspect-medalpaca/llama-3.3-7b-instruct" \
    --data_path "$data_path" \
    --output_dir "$output_path" \
    --train_in_8bit False \
    --use_lora False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --gradient_checkpointing True \
    --global_batch_size 128 \
    --per_device_batch_size 4 \
    --num_epochs 5
