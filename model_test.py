import argparse
import json
import os.path
import os

from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pandas as pd
import openai
import random
import time
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def interactive_predict(args):
    """
    交互式体验
    :return:
    """
    while True:
        try:
            input_text = input('I:')
            tokenizer, model = load_model(args.pretrained_model_path, args.lora_path)
            output_text,_,_,_,_,_ = predict_and_tokenize(model, tokenizer, [input_text])
            print('O:', output_text)
        except KeyboardInterrupt:
            return

def load_model(pretrain_model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, 
                                            # padding_side='left', 
                                            trust_remote_code=True, 
                                            use_fast=False,
                                            model_max_length=512) 
    # tokenizer.pad_token_id = 0
    logger.info("Start loading model: %s", pretrain_model_path)
    
    model = AutoModelForCausalLM.from_pretrained(pretrain_model_path,
                                                device_map="auto",
                                                trust_remote_code=True)
    
    # Check if LoRA weights are provided
    if lora_path:
        logger.info("Applying LoRA fine-tuned weights from: %s", lora_path)
        try:
            model = PeftModel.from_pretrained(model, lora_path)
            logger.info("LoRA fine-tuned weights successfully applied.")
        except Exception as e:
            logger.error("Failed to apply LoRA weights: %s", str(e))
            raise e

    logger.info("Model loading completed")
    
    model = model.eval()
    return tokenizer, model

def promptor(item, mode):
    if mode == 'instruction':
        prompt = f"对以下{item['domain']}领域的文本进行纠错。注意，{item['instruction']}\n请直接给出答案，不要添加前缀词。\n输入: {item['input']}\n输出: "
    elif mode == 'non_instruction':
        prompt = f"请改正输入文本中的错别字。如果错别字不存在，直接输出原本输入。\n输入: {item['input']}\n输出: "
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'instruction' or 'non_instruction'.")
    return prompt

    if template['template_name'] == "prediction_extract":
        response['predict'] = re.sub(r'\n\n输入.*', '', response['predict'], flags=re.DOTALL)
        user_content = template['user_content'].format(
            source_sentence=response['input'],
            instance_index=response['instance_index'],
            prediction=response['predict']
        )

def predict_and_tokenize(model, tokenizer, item):
    """
    预测一个query
    :return:
    """
    texts = [item]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    
    inputs = inputs.to(model.device)
    max_seq_len = inputs['input_ids'].shape[1]
    
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=300,
        num_beams=1
    )
    outputs_pre = outputs[:, max_seq_len:]
    outputs_pre = tokenizer.batch_decode(outputs_pre, skip_special_tokens=True)
    
    outputs_ids = outputs
    outputs_token = [tokenizer.decode(d) for d in outputs_ids[0]]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs_tokenizer_ids = tokenizer(outputs[0])['input_ids']
    outputs_tokenizer_token = [tokenizer.decode(d) for d in outputs_tokenizer_ids]
    flag = bool(outputs_ids[0].tolist() == outputs_tokenizer_ids) or bool(outputs_ids[0].tolist()[:-1] == outputs_tokenizer_ids)
    
    return outputs_pre[0], outputs_ids[0].tolist(), outputs_token, outputs_tokenizer_ids, outputs_tokenizer_token, flag

        
def batch_predict(args):
    """
    批量预测
    :return:
    """
    tokenizer, model = load_model(args.pretrained_model_path, args.lora_path)
    data = json.load(open(args.testset_dir, 'rt', encoding='utf-8')) # data is a list of testset items
    data_with_predict = []
    
    # Save the model response in lines in case of crashing.
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    with open(args.output_dir, 'a', encoding='utf-8') as f:
        for item in tqdm(data):
            prompt = promptor(item, args.mode)
            item['predict'], _, _, _, _, _ = predict_and_tokenize(model, tokenizer, prompt)

            data_with_predict.append(item)
            print(prompt)
            print('-' * 20)
            print(item['predict'])
            print('=' * 20)

            json.dump(item, f, ensure_ascii=False)
            f.write('\n')  # jsonl 포맷이므로 각 json 객체를 한 줄로 기록


    
    json.dump(data_with_predict, open(args.output_dir, 'wt', encoding='utf-8'),
            ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--lora_path", type=str, help="Path to the checkpoint. If non, test with vanila model.")
    parser.add_argument("--testset_dir", type=str)
    parser.add_argument("--output_dir", type=str, help="Path to store the results.")
    parser.add_argument("--mode", type=str, default="batch", help="interaction or batch")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    #interactive_predict(args)
    batch_predict(args)