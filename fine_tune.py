import os
import math
import pathlib
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from transformers.training_args import TrainingArguments
import pdb
from transformers.file_utils import TRANSFORMERS_CACHE
from utils.calculate_metric import compute
import logging
import ipdb
from collections import Counter
from utils.calculate_metric import get_label
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="baichuan-inc/Baichuan2-7B-Base")


@dataclass
class DataArguments:
    train_dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_dataset_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    evaluation_strategy: str = field(
        default="epoch",  # 또는 "steps"로 변경 가능
        metadata={"help": "The evaluation strategy to use. 'steps' or 'epoch'."}
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        user_tokens=[195],
        assistant_tokens=[196],
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = []
        labels = []

        # Prompting
        prompt = f"以下是{example['domain']}领域中输入的文本。请改正输入文本中的错别字，然后直接输出改正后的文本。如果错别字不存在，直接输出原本输入。注意，{example['instruction']}\n输入: {example['input']}\n输出: "
        response = example['output']
        
        # Tokenizing
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        
        # 사람 메시지 (prompt)에 대해 레이블 무시 (ignore_index)
        input_ids += self.user_tokens + prompt_ids
        labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(
                    prompt_ids
                )
        
        # 어시스턴트의 응답 (response)에 대한 레이블 설정
        input_ids += self.assistant_tokens + response_ids
        labels += [self.ignore_index] + response_ids
        
        # EOS 토큰 추가 (응답의 끝에만)
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        
        # 모델의 최대 길이에 맞춰 자르고, 패딩 추가
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        
        
        # LongTensor로 변환
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        
        # Attention Mask 생성 (패딩이 아닌 부분만 1로 설정)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    


    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.info(f"Loading model: {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    logging.info(f"Model loaded: {model_args.model_name_or_path}")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length
    )
    if training_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        if 'Baichuan' in model_args.model_name_or_path:
            logging.info("Applying LoRA config for Baichuan. Model: %s", model_args.model_name_or_path)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["W_pack"],
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
        else:
            logging.info("Applying LoRA config for Qwen. Model: %s", model_args.model_name_or_path)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                target_modules=["q_proj","k_proj","v_proj"],
                lora_dropout=0.1,
            )

        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_dataset = SupervisedDataset(
        data_args.train_dataset_path, tokenizer, training_args.model_max_length
    )
    eval_dataset = SupervisedDataset(
        data_args.eval_dataset_path, tokenizer, training_args.model_max_length
    )
    logging.info("Training dataset loaded with %d samples.", len(train_dataset))
    logging.info("Evaluation dataset loaded with %d samples.", len(eval_dataset))

    
    
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        inputs = eval_pred.inputs
        
        predictions = np.argmax(predictions, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and labels (target sentences)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        
        decoded_inputs = [
            decoded_input.split('\n输入:')[1].split('\n输出:')[0].strip()
            for decoded_input in decoded_inputs
            if '\n输入:' in decoded_input and '\n输出:' in decoded_input
        ]
        
        wrong = 0
        context_mismatch = 0
        general_csc_err = 0
        # Now you have the source sentences and can combine them for metric calculation
        for input, label, prediction in zip(decoded_inputs, decoded_labels, decoded_predictions):
            # input = input.split("输入：", 1)
            label, err_types,_ = get_label(input, label, prediction, None, 2)
            if not label:
                wrong += 1
                for err_type in err_types:
                    if err_type == "context_mismatch":
                        context_mismatch += 1
                    elif err_type == "general_csc_err":
                        general_csc_err += 1

        return {
            "context_mismatch": context_mismatch / len(prediction) * 100,
            "general_csc_err": general_csc_err / len(prediction) * 100,
            "accuracy": wrong / len(prediction) * 100 
        }
        
        # predictions = p.predictions.argmax(-1)
        # pred_sentences = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # # Use only the eval_dataset data, not train_dataset
        # src_sents, trg_sents, keywords, domains, instructions, index = [], [], [], [], [], []
        # for item in eval_dataset.data:  # This should be eval_dataset, not train_dataset
        #     src_sents.append(item["input"])
        #     trg_sents.append(item["output"])
        #     keywords.append(item["typo"])
        #     domains.append(item["domain"])
        #     instructions.append(item["instruction"])
        #     index.append(item["index"])

        # # Now you can compute your metrics for the eval dataset
        # _, _, f1, _, _, _, _, wrong_sents = compute(src_sents, trg_sents, pred_sentences, keywords, domains, instructions, index, "token", 2)
        # err_types_counts = Counter(err_type for item in wrong_sents for err_type in item['err_type'])
        
        # # Save counts for each error type
        # non_keyword_count = err_types_counts.get("non_keyword", 0)
        # context_mismatch_count = err_types_counts.get("context_mismatch", 0)
        # general_csc_err_count = err_types_counts.get("general_csc_err", 0)

        # # Calculate the percentages for each error type
        # non_keyword_percentage = non_keyword_count / len(wrong_sents) * 100
        # context_mismatch_percentage = context_mismatch_count / len(wrong_sents) * 100
        # general_csc_err_percentage = general_csc_err_count / len(wrong_sents) * 100

        # return {
        #     "f1": f1,
        #     "non_keyword": non_keyword_percentage,
        #     "context_mismatch": context_mismatch_percentage,
        #     "general_csc_err": general_csc_err_percentage
        # }

    trainer = transformers.Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    logging.info("Training completed. Saving directory: %s", training_args.output_dir)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
