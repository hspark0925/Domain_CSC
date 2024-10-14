from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import HfArgumentParser, TrainingArguments, AutoModelForCausalLM, Trainer, AutoTokenizer, TrainerCallback
from llm_train_util import prepare_model_for_training, SFTDataCollector
import logging
import os
import bitsandbytes
import ninja

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def print_model_statistics(model):
    """
    Prints the total number of parameters (trainable and non-trainable) in the given model 
    and estimates the GPU memory required to store those parameters.

    Parameters:
    model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
    None
    """
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate the total memory required by all parameters in bytes
    memory_in_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Convert memory to gigabytes
    memory_in_gb = memory_in_bytes / 1e9

    # Print the results
    print(f"Model Statistics:")
    print(f"- Total Parameters: {total_params:,}")
    print(f"- Trainable Parameters: {trainable_params:,}")
    print(f"- Estimated GPU Memory: {memory_in_gb:.2f} GB")

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

@dataclass
class FinetuneArguments:
    """
    微调参数
    """
    pretrained_model_path: str = field()
    train_dataset_path: str = field()
    eval_dataset_path: str = field()
    pad_token_id: int = field(default=0)
    lora_rank: int = field(default=16)
    lora_alpha: float = field(default=32.0)
    lora_dropout: float = field(default=0.1)
    lora_target: str = field(default="W_pack")
    ft_type: str = field(default="lora")


def create_and_prepare_dataset(data_path):
    """
    创建数据
    :return:
    """
    train_dataset = load_dataset("json", data_files=data_path)

    def preprocess_function(example):
        """
        预处理
        :param example:
        :return:
        """
        prompt = f"对以下{example['domain']}领域的文本进行纠错。注意，{example['instruction']}\n请直接给出答案，不要添加前缀词。\n输入: {example['input']}\n输出: "
        response = example['output']
        return {
            'prompt': prompt,
            'response': response,
        }

    train_dataset = train_dataset.map(preprocess_function, batched=False)
    return train_dataset['train']


def train():
    """
    训练模型
    :return:
    """
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)).parse_args_into_dataclasses()
    # 
    # load model
    logger.info("Start loading model: %s", finetune_args.pretrained_model_path)
    if 'Baichuan' in finetune_args.pretrained_model_path:       
        model = AutoModelForCausalLM.from_pretrained(finetune_args.pretrained_model_path,
            revision="v2.0",
            trust_remote_code=True)
        
        if finetune_args.ft_type == 'lora':
            model = prepare_model_for_training(model)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetune_args.lora_rank,
                lora_alpha=finetune_args.lora_alpha,
                lora_dropout=finetune_args.lora_dropout,
                target_modules=["W_pack"]
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(finetune_args.pretrained_model_path,
                                                    trust_remote_code=True)
        if finetune_args.ft_type == 'lora':
            model = prepare_model_for_training(model)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetune_args.lora_rank,
                lora_alpha=finetune_args.lora_alpha,
                lora_dropout=finetune_args.lora_dropout,
                target_modules=[target.strip() for target in finetune_args.lora_target.split(",")]
            )
    logger.info("Loading Completed")
    print_model_statistics(model)
    
    model = get_peft_model(model, lora_config)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_args.pretrained_model_path, trust_remote_code=True, use_fast=False)
    if 'Baichuan' in finetune_args.pretrained_model_path:
        tokenizer.pad_token_id = 0
    elif 'Qwen' in finetune_args.pretrained_model_path:
        tokenizer.pad_token_id = 135269
        tokenizer.eos_token_id = 135269
    # load dataset
    train_dataset = create_and_prepare_dataset(finetune_args.train_dataset_path)
    eval_dataset = create_and_prepare_dataset(finetune_args.eval_dataset_path)
    # start train
    training_args.ddp_find_unused_parameters = True # False
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=SFTDataCollector(tokenizer),
        callbacks=[PeftSavingCallback()] if finetune_args.ft_type == 'lora' else None,
    )
    trainer.train()

if __name__ == '__main__':
    train()
