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
from calculate_metric import compute

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

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="baichuan")
    parser.add_argument("--results_dir", type=str, default="./dataset/results/")
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    result_path = os.path.join(args.results_dir, args.pretrained_model_path, "inference_result.json")
    data = processor._read(result_path)
    
    src_sents, trg_sents, prd_sents, keywords, domains, instructions = [], [], [], [], [], []
    for line in data:
        clean_input = re.sub(r'\s+', '', line["input"])
        clean_output = re.sub(r'\s+', '', line["output"])
        clean_prediction = re.sub(r'\s+', '', line["extracted_prediction"])
        
        src_sents.append(clean_input)
        trg_sents.append(clean_output)
        prd_sents.append(clean_prediction)
        domains.append(line["domain"])
        instructions.append(line["instruction"])
        keywords.append(line["keyword"][1])
    
    p, r, f1, fpr, tp_sents, fp_sents, fn_sents = compute(src_sents, trg_sents, prd_sents, keywords, domains, instructions, k=2)
    
    result = {
        "p": round(p * 100, 2),
        "r": round(r * 100, 2),
        "f1": round(f1 * 100, 2),
        "fpr": round(fpr * 100, 2),
    }

    # Log the evaluation results
    logger.info("***** Eval results *****")
    logger.info(f"Model: {args.pretrained_model_path}")
    logger.info(f"{result}")
    
    processor._write(os.path.join(args.results_dir, args.pretrained_model_path, "fp_sents.json"), fp_sents)
    processor._write(os.path.join(args.results_dir, args.pretrained_model_path, "fn_sents.json"), fn_sents)


if __name__ == "__main__":
    main()