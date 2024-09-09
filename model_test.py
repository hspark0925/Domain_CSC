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

def load_model(pretrain_model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, 
                                              padding_side='left', 
                                              trust_remote_code=True, 
                                              use_fast=False) 
    tokenizer.pad_token_id = 0
    logger.info("Start loading model: %s", pretrain_model_path)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrain_model_path,
        device_map="auto",
        # torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    # model = model.merge_and_unload()
    logger.info("Loading Completed")
    
    model = model.eval()
    # model = model.cuda()
    return tokenizer, model

def predict_samples(model, tokenizer, item):
    """
    预测一个query
    :return:
    """
    texts = [f"对以下{item['domain']}领域的文本进行纠错。注意，{item['instruction']}\n请直接给出答案，不要添加前缀词。\n输入: {item['input']}\n输出: "]
    
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

        
def batch_predict(pretrained_model_path, lora_path, args):
    """
    批量预测
    :return:
    """
    tokenizer, model = load_model(pretrained_model_path, lora_path)
    data = json.load(open(f'{args.dataset_dir}instruction_with_instances.json', 'rt', encoding='utf-8'))
    data_with_predict = []
    for item in tqdm(data):
        item['predict'],_,_,_,_, item['flag'] = predict_samples(model, tokenizer, item)
        
        data_with_predict.append(item)
        print(f"任务: 纠错文本\n输入: {item['input']}\n输出: ")
        print('-' * 20)
        print(item['predict'])
        print('=' * 20)

    file_path = os.path.join(args.dataset_dir, 'results', os.path.basename(pretrained_model_path), 'inference_result.json')

    # 디렉토리 생성
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    json.dump(data_with_predict, open(file_path, 'wt', encoding='utf-8'),
            ensure_ascii=False, indent=4)
   
        
def interactive_predict(args):
    """
    交互式体验
    :return:
    """
    while True:
        try:
            input_text = input('I:')
            tokenizer, model = load_model(args.pretrained_model_path, args.lora_path)
            output_text,_,_,_,_,_ = predict_samples(model, tokenizer, [input_text])
            print('O:', output_text)
        except KeyboardInterrupt:
            return

def make_prompt(args):
    """
    构造few-shot prompts
    :return:
    """
    dir = args.dataset_dir + "train_data/cscd_train.json"
    true_num, false_num = 2, 2
    prompts = {}
    with open(dir, "r", encoding="utf-8") as fh:
        Data = json.load(fh)

    length_dict = {"cscd": 5000}

    for k in length_dict.keys():

        for i in range(length_dict[k]):
            prompt = "纠正句子中的错别字，并返回纠正后的句子。\n\n"
            t_n, f_n = 0 ,0
            index_list = []
            flag_true, flag_false = False, False

            while True:
                if flag_false and flag_true: break
                randint = random.randint(0, len(Data)-1)
                if Data[randint]['src'] == Data[randint]['tgt'] and not flag_true and randint not in index_list: 
                    t_n += 1
                    index_list.append(randint)
                if Data[randint]['src'] != Data[randint]['tgt'] and not flag_false and randint not in index_list: 
                    f_n += 1
                    index_list.append(randint)
                if t_n == true_num: flag_true = True
                if f_n == false_num: flag_false = True
            
            assert len(index_list) == true_num + false_num, len(index_list)
            random.shuffle(index_list)
            for index in index_list:
                if args.mode =='new': prompt += " ".join(list(Data[index]['src'])) + "=>" + " ".join(list(Data[index]['tgt'])) + '\n'
                if args.mode =='old': prompt += Data[index]['src'] + "=>" + Data[index]['tgt'] + '\n'
            prompts[i] = prompt

        json.dump([prompts], open((args.dataset_dir+f'prompt/prompt_{k}_{args.mode}.json'), 'wt', encoding='utf-8'),
                    ensure_ascii=False, indent=4)

              
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default='./dataset/')
    parser.add_argument("--mode", type=str, default='old')
    parser.add_argument("--base_dir", type=str, default="/predicts/prediction_qwen_14b/all_result/") 
    parser.add_argument("--domain", type=str, default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    #interactive_predict(args)
    batch_predict(args.pretrained_model_path, args.lora_path, args)