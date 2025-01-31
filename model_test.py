import argparse
import json
import os.path
import os
import sys
import ipdb

from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch
import pandas as pd
import openai
import random
import time
import logging
from utils.prompter import Prompter
import re
from evaluate_result import evaluate

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def interactive_predict(args):
    """
    交互式体验
    :return:
    """
    tokenizer, model = load_model(args.pretrained_model_path, args.lora_path)
    while True:
        try:
            print("I:")
            input_text = sys.stdin.read()
            if input_text.strip() == "exit":
                return
            
            messages = []
            messages.append({"role": "system", "content": "你是一位精通中文的人，对中文的各个领域都有深入的理解。"})
            messages.append({"role": "user", "content": input_text})
            output_text = predict_and_tokenize(model, tokenizer, messages, args.pretrained_model_path, args.test_mode)
            print('O:', output_text)
        except KeyboardInterrupt:
            return

def load_model(pretrain_model_path, lora_path):
    if "gpt" in pretrain_model_path.lower():
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, 
                                        # padding_side='left', 
                                        trust_remote_code=True, 
                                        use_fast=False,
                                        model_max_length=1024) 
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

def predict_and_tokenize(model, tokenizer, messages: list[dict], model_path, test_mode):
    """
    预测一个query
    :return:
    """
    if "qwen" in model_path.lower():
        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
        model_inputs = tokenizer(
            [text], 
            return_tensors="pt",
            padding=True,
            truncation=True,
            ).to(model.device) 
        
        if test_mode == "false_case_generation":
            err_keyword_match = re.search(r"关键词:\s*(.+)", messages[-1]['content'])
            if not err_keyword_match:
                raise ValueError("Couldn't find the keyword in this content: {}".format(messages[-1]['content']))
            err_keyword = err_keyword_match.group(1).strip()
            
            generated_ids = []
            num_return_sequences = 5
            
            outputs = []
            for i in range(num_return_sequences):
                temp = 1 + i * 0.2
                top_k_value = 50 + i * 25
                top_p_value = 0.8 + i * 0.05
                
                generated_ids = model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    top_k=top_k_value,
                    top_p=top_p_value,
                    temperature=temp,
                    do_sample=True,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                outputs.append(output[0])
                if err_keyword.lower() not in output[0].lower():
                    break
                else:
                    continue
            return outputs
        else:
            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return output

    elif "baichuan" in model_path:
        outputs = model.chat(
            tokenizer, 
            messages
            )
        return [outputs]
    elif "gpt" in model_path:
        try:
            response = openai.ChatCompletion.create(
                model=model_path,
                messages=messages,
            )
            responses = [choice.message['content'] for choice in response.choices]
            return responses
        
        except openai.error.RateLimitError as e:
            if attempt < retries - 1:  # Only wait and retry if we have retries left
                logger.info(f"Rate limit reached. Attempt {attempt + 1} of {retries}. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logger.info("Rate limit reached and all retries exhausted.")
            
    else:
        raise ValueError("Unknown model path for chat template: %s" % model_path)

        
def batch_predict(args):
    """
    批量预测
    :return:
    """
    tokenizer, model = load_model(args.pretrained_model_path, args.lora_path)
    
    data = json.load(open(args.testset_dir, 'rt', encoding='utf-8'))
    
    # Configurations for false_case_generation.
    if args.test_mode == "false_case_generation":
        data = [item for item in data if item['keyword_label'] == 1]
    
    data_with_predict = []
    os.makedirs(os.path.dirname(args.output_dir + ".jsonl"), exist_ok=True)
    with open(args.output_dir+".jsonl", "w") as file:
        pass
    
    with open(args.output_dir + ".jsonl", 'a', encoding='utf-8') as f:
        prompter = Prompter(args.template_dir, args.testset_dir, args.test_mode, verbose=True)
        
        logger.info("****** Messages example ******")
        logger.info("\n------------------\n".join([message['content'] for message in prompter.gen_messages(data[0], args.incontext_learning)]))
        logger.info("****** Starting Test ******")
        
        for item in tqdm(data):
            messages = prompter.gen_messages(item, args.incontext_learning)
            item['predict'] = predict_and_tokenize(model, tokenizer, messages, args.pretrained_model_path, args.test_mode)
            

            data_with_predict.append(item)
            print(messages[-1]['content'])
            print('-' * 20)
            print(item['predict'][-1])
            print('=' * 20)
            
            if args.test_mode != "false_case_generation":
                item['predict'] = item['predict'][0]
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    os.makedirs(os.path.dirname(args.output_dir + ".json"), exist_ok=True)
    json.dump(data_with_predict, open(args.output_dir + ".json", 'wt', encoding='utf-8'),
            ensure_ascii=False, indent=4)
    logger.info(f"Results saved to {args.output_dir}.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--lora_path", type=str, help="Path to the checkpoint. If none, test with vanila model.")
    parser.add_argument("--testset_dir", type=str)
    parser.add_argument("--output_dir", type=str, help="Path to store the results.")
    parser.add_argument("--test_mode", type=str, default="instruction", help="interaction, non_instruction, interaction, false_case_generation")
    parser.add_argument("--incontext_learning", type=int, default=3, help="Numbers of shots.")
    parser.add_argument("--template_dir", type=str, default="inference", help="Prompt template name.")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    if args.test_mode == "interaction":
        interactive_predict(args)
    else:
        batch_predict(args)
        evaluate(args.output_dir)
        