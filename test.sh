BASE_DIR=$PWD
MODEL_PATH="baichuan-inc/Baichuan2-13B-Chat"

gpus=(0 1 2 3)
gpu_index= 0

#LORA_DIR=$BASE_DIR/checkpoint/old_tokenizer/xxxx/checkpoint-900
LORA_DIR=$

CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python model_test.py --pretrained_model_path $MODEL_PATH --lora_path=$LORA_DIR &