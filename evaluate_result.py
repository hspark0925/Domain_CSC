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
from utils.calculate_metric import keyword_compute, sent_compute
from collections import Counter
import ipdb

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO,
                    filename='temp.log')
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
    text = text.lower()
    return text

def prediction_process(prediction, input):
    prediction = re.sub(r'\n\n输入.*', '', prediction, flags=re.DOTALL)
    prediction = clean_text(prediction)
    
    #Copy input
    copy_input = ["正确", "无误", "正确。", "正确输入", "无需修改", "错误:无", "错别字"]
    
    if len(prediction) < 25 and any(word in prediction for word in copy_input):
        return input
    
    #Prefix Removal
    prefix = ["正确输入:", "正确版本:", "错误:", "错误纠正后的句子:", "错误纠正后的文本:", "修正后:", "纠正后:", "修正后的文本: ", "以下是修改后的文本:"]
    for p in prefix:
        if prediction.startswith(p):
            return prediction.removeprefix(p)
        
    return prediction
    
   
def evaluate(results_dir):
    
    processor = DataProcessor()
    
    data = processor._read(results_dir)
    src_sents, trg_sents, prd_sents, keywords, domains, instructions, index = [], [], [], [], [], [], []
    for line in data:        
        src_sents.append(clean_text(line["input"]))
        trg_sents.append(clean_text(line["output"]))
        
        prd_sents.append(prediction_process(line["predict"], clean_text(line["input"])))
        domains.append(line["domain"])
        instructions.append(line["instruction"])
        keywords.append([clean_text(keyword) for keyword in line["keyword"]])
        index.append(line['instance_index'])
    
    # Sentence Level Evaluation
    p, r, f1, fpr, tp_sents, fp_sents, fn_sents, wrong_sents = sent_compute(src_sents, trg_sents, prd_sents, keywords, domains, instructions, index, k=2)
    
    result = {
        "p": round(p * 100, 1),
        "r": round(r * 100, 1),
        "f1": round(f1 * 100, 1),
        "fpr": round(fpr * 100, 1),
    }

    logger.info(f"Test Mode: {os.path.splitext(results_dir)[0]}")
    
    # Log the evaluation results
    logger.info("***** Sentence Level results *****")
    logger.info(f"{result}")
    
    logger.info(f"Accuracy: {(1-len(wrong_sents)/len(data)) * 100:.1f}%")
    
    # Keyword Level Evaluation
    p, r, f1, fpr, tp_sents, fp_sents, fn_sents, wrong_sents = keyword_compute(src_sents, trg_sents, prd_sents, keywords, domains, instructions, index, k=2)
    
    result = {
        "p": round(p * 100, 1),
        "r": round(r * 100, 1),
        "f1": round(f1 * 100, 1),
        "fpr": round(fpr * 100, 1),
    }

    # Log the evaluation results
    logger.info("***** Keyword Level results *****")
    logger.info(f"{result}")
    
        
    err_types_counts = Counter(err_type for item in wrong_sents for err_type in item['err_type'])
    logger.info(f"Accuracy: {(1-len(wrong_sents)/len(data)) * 100:.1f}%")
    for value_type, count in err_types_counts.items():
        logger.info(f"{value_type}: {count/len(data) * 100:.1f}%")
    
    processor._write(os.path.join(os.path.dirname(results_dir), "fp_sents.json"), fp_sents)
    processor._write(os.path.join(os.path.dirname(results_dir), "fn_sents.json"), fn_sents)
    processor._write(os.path.join(os.path.dirname(results_dir), "wrong_sents.json"), wrong_sents)
    processor._write

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="result/01-08/qwen_contrastive/qwen_contrastive_result.json")
    args = parser.parse_args()
    evaluate(args.results_dir)

if __name__ == "__main__":
    main()