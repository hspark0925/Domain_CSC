#!/bin/bash

model="qwen"  # qwen or baichuan
LORA_DIR="" # path to finetune weights, if none run with base model
test_mode="contrastive" #mode instruction or non_instruction
gpu_index=(0 1 2 3)
incontext_learning=3
test_set="./dataset/dcsc.json"

#cached model path/home/piaoxx/.cache/huggingface/hub/models--baichuan-inc--Baichuan2-13B-Chat/snapshots/c8d877c7ca596d9aeff429d43bff06e288684f45
BAICHUAN_2_13B_CHAT_PATH="/home/piaoxx/.cache/huggingface/hub/models--baichuan-inc--Baichuan2-13B-Chat/snapshots/c8d877c7ca596d9aeff429d43bff06e288684f45"
QWEN_2_7B_INSTRUCT_PATH="/home/piaoxx/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"


DATE=$(date '+%m-%d')
if [ "$model" = "qwen" ]; then
    MODEL_PATH=$QWEN_2_7B_INSTRUCT_PATH
else
    MODEL_PATH=$BAICHUAN_2_13B_CHAT_PATH
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
