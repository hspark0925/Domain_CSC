#!/bin/bash

model="Qwen/Qwen2.5-0.5B-Instruct"  # qwen or baichuan
LORA_DIR="" # path to finetune weights, if none run with base model
test_mode="contrastive" #mode instruction or non_instruction
gpu_index=(0 1 2 3)
incontext_learning=3
test_set="./dataset/dcsc.json"


DATE=$(date '+%m-%d')
if [ "$model" = "gpt" ]; then
    MODEL_PATH="gpt"
else
    MODEL_PATH=$model
fi

if [ -n "$LORA_DIR" ]; then #if finetune model
    OUTPUT_PATH="./result/"$DATE"/"$model"_FT_"$test_mode"_"$incontext_learning"/"$model"_FT_"$test_mode"_result"
else #if base model
    OUTPUT_PATH="./result/"$DATE"/"$model"_"$test_mode"_"$incontext_learning"/"$model"_"$test_mode"_result"
fi

CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpu_index[*]}") python model_test.py \
    --pretrained_model_path "$MODEL_PATH" \
    --lora_path "$LORA_DIR" \
    --testset_dir "$test_set" \
    --output_dir $OUTPUT_PATH \
    --test_mode $test_mode \
    --incontext_learning $incontext_learning \
    --template "inference" \
