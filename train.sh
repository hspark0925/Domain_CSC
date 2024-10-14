set -ex
export WANDB_API_KEY=11218807886bb5d12911917dbf678c2484bc3d85
export WANDB_PROJECT=domain_csc
export CUDA_VISIBLE_DEVICES=0,1,2,3

BASE_DIR=$PWD
DATE=$(TZ=Asia/Shanghai date +'%Y%m%d%H%M%S')


BAICHUAN_2_13B_CHAT_PATH="/home/piaoxx/.cache/huggingface/hub/models--baichuan-inc--Baichuan2-13B-Chat/snapshots/c8d877c7ca596d9aeff429d43bff06e288684f45"
QWEN_2_7B_INSTRUCT_PATH="/home/piaoxx/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"
LORA_TARGET=q_proj,k_proj,v_proj


export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$BASE_DIR/models/common
export HOME=/tmp/home
export CUDA_HOME=/home/piaoxx/miniconda3/envs/env

DATA_DIR=$BASE_DIR/dataset

# MODEL_PATH=$QWEN_2_7B_INSTRUCT_PATH
MODEL_PATH=$QWEN_2_7B_INSTRUCT_PATH
LORA_TARGET=$LORA_TARGET

FT_TYPE=lora
RUN_NAME=domain-csc-$DATE
OUTPUT_DIR=$BASE_DIR/checkpoint/$RUN_NAME


MASTER_PORT=$(shuf -n 1 -i 10000-65535)
torchrun --nproc_per_node=4 --master_port $MASTER_PORT fine-tune.py \
    --do_eval \
    --do_train \
    --train_dataset_path $DATA_DIR/dcsc_train.json \
    --eval_dataset_path $DATA_DIR/dcsc_dev.json \
    --report_to "none" \
    --model_name_or_path $MODEL_PATH \
    --output_dir "output" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --use_lora True \
    --logging_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --save_strategy steps \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --prediction_loss_only \
    --save_total_limit 25 \
    --num_train_epochs 50 \
    --dataloader_num_workers 4 \
    --log_level info \
    --remove_unused_columns false \
    --output_dir $OUTPUT_DIR \
    --report wandb \
    --run_name $RUN_NAME 
    # --tf32 True
    # --bf16 True \

    # --do_eval \
    # --do_train \
    # --pretrained_model_path $MODEL_PATH \
    # --train_dataset_path $DATA_DIR/dcsc_train.json \
    # --eval_dataset_path $DATA_DIR/dcsc_dev.json \
    # --per_device_train_batch_size 1 \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 32 \
    # --lora_target $LORA_TARGET \
    # --logging_steps 4 \
    # --eval_steps 100 \
    # --save_steps 100 \
    # --load_best_model_at_end \
    # --evaluation_strategy steps \
    # --save_strategy steps \
    # --metric_for_best_model eval_loss \
    # --greater_is_better false \
    # --prediction_loss_only \
    # --save_total_limit 25 \
    # --num_train_epochs 50 \
    # --dataloader_num_workers 4 \
    # --log_level info \
    # --remove_unused_columns false \
    # --output_dir $OUTPUT_DIR \
    # --report wandb \
    # --run_name $RUN_NAME \
    # --ft_type $FT_TYPE \
    # --seed 3407 \
    # --deepspeed $BASE_DIR/ds_config/ds_config.json \
    # --learning_rate 1e-4 
    # # --adam_beta1 0.9 \
    # # --adam_beta2 0.98 \
    # # --adam_epsilon 1e-8 \
    # # --max_grad_norm 1.0 \
    # # --weight_decay 1e-4 \
    # # --learning_rate 2e-5
    # # --learning_rate 1e-4 \
    # # --fp16 \
    # # --fp16_full_eval \