from __future__ import absolute_import, division, print_function
import argparse
import logging
import glob
import os
import random
import math
import copy
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
import re
from utils.calculate_metric import compute
from collections import Counter
import ipdb

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processor for the data set:
    """
    @staticmethod
    def _read(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    
    def _write(self, output_file, data):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def clean_text(text):
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'：', ':', text)
    text = re.sub(r'，', ',', text)
    text = re.sub(r'！', '!', text)
    return text

def prediction_process(prediction, input):
    prediction = re.sub(r'\n\n输入.*', '', prediction, flags=re.DOTALL)
    prediction = clean_text(prediction)
    
    #Copy input
    copy_input = ["正确", "无误", "正确。", "正确输入", "无需修改", "错误:无", "错别字"]
    
    if len(prediction) < 25 and any(word in prediction for word in copy_input):
        return input
    
    #Prefix Removal
    prefix = ["正确输入:", "正确版本:", "错误:", "错误纠正后的句子:", "错误纠正后的文本:", "修正后:", "纠正后:"]
    for p in prefix:
        if prediction.startswith(p):
            return prediction.removeprefix(p)
        
    return prediction
    
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./result/11-28/qwen_instruction/qwen_instruction_result.json")
    parser.add_argument("--metric_mode", type=str, default="token",help="token or sentence")
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    data = processor._read(args.results_dir)
    src_sents, trg_sents, prd_sents, keywords, domains, instructions, index = [], [], [], [], [], [], []
    for line in data:        
        src_sents.append(clean_text(line["input"]))
        trg_sents.append(clean_text(line["output"]))
        prd_sents.append(prediction_process(line["predict"], clean_text(line["input"])))
        domains.append(line["domain"])
        instructions.append(line["instruction"])
        keywords.append([clean_text(keyword) for keyword in line["typo"]])
        index.append(line['instance_index'])
    
    p, r, f1, fpr, tp_sents, fp_sents, fn_sents, wrong_sents = compute(src_sents, trg_sents, prd_sents, keywords, domains, instructions, index, args.metric_mode, k=2)
    
    result = {
        "p": round(p * 100, 2),
        "r": round(r * 100, 2),
        "f1": round(f1 * 100, 2),
        "fpr": round(fpr * 100, 2),
    }

    # Log the evaluation results
    logger.info("***** Eval results *****")
    logger.info(f"Test mode: {os.path.splitext(args.results_dir)[0]}")
    logger.info(f"{result}")
    
    err_types_counts = Counter(err_type for item in wrong_sents for err_type in item['err_type'])
    for value_type, count in err_types_counts.items():
        logger.info(f"{value_type}: {count/len(wrong_sents) * 100:.2f}%")
    
    processor._write(os.path.join(os.path.dirname(args.results_dir), "fp_sents.json"), fp_sents)
    processor._write(os.path.join(os.path.dirname(args.results_dir), "fn_sents.json"), fn_sents)
    processor._write(os.path.join(os.path.dirname(args.results_dir), "wrong_sents.json"), wrong_sents)



if __name__ == "__main__":
    main()