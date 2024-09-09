from collections import defaultdict, Counter
import argparse
import logging
import os
import json
from tqdm.auto import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processor for the dataset
    """
    
    def get_source_data(self, data_dir, domain_type, filename):
        return self._read(os.path.join(data_dir, domain_type, filename))
    def get_instruction_data(self, data_dir, filename):
        return self._read(os.path.join(data_dir, filename))
    
    @staticmethod
    def _read(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def get_domains_from_instruction(self, domains_list):
        """
        Return teh sorted list of domains.
        """
        sorted_domains = sorted(Counter(domains_list).items(), key=lambda x: x[1], reverse=True)
        sorted_domain_list = [domain for domain, count in sorted_domains]
        
        return sorted_domain_list
    
    def user_domain_selection(self, source_domain, domains):
        print("Source Domain: ", {source_domain})
        # print("List of Instruction Domains: ")
        # print("; ".join(f"{i}: {domain}" for i, domain in enumerate(domains)))
        
        while True:
            try:
                user_input = input(f"Enter your choice(s) (space-separated numbers, e.g., 1 3 4): ")
                choices = [int(i) for i in user_input.split(" ")]
                if all(0 <= choice < len(domains) for choice in choices):
                    print("Selected Domains: ",  [domains[i] for i in choices])
                    return choices                      
                else:
                    print("Some choices are out of range. Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
        
    
    

class Extract:
    def __init__(self, source_data):
        """
        Initializes the class and loads the source data.
        source_data: List of sentences.
        """
        self.source_data = source_data  # List of source sentences
        
    def extraction(self, keyword):
        """
        Extracts the sentences that contain the specified keyword.
        keyword: The keyword to search for.
        return: List of actual sentences containing the keyword.
        """   
        result = []
        for instance in self.source_data:
            if keyword in instance['trg']:
                result.append(instance)
            
            if len(result) >= 2:
                break
        return result
    
    def create_label_and_source(self, err, fix, instance):
        """
        If the error and the fixed keyword is differnt, replace the fixed keyword to the error in the sentence.
        return: label of the instance, source sentence.
        """
        if err != fix:
            src = instance['src'].replace(fix, err)
            trg = instance['trg']
            return 1, src, trg #labeled 1, correction needed.
        else:
            src, trg = instance['src'], instance['trg']
            return 0, src, trg #The source and the target is the same. No correction needed, labeled 0.
    
    def extract_from_source(self, instruction, file):
        """
        Processes an instruction and returns a dictionary for extracted or non-extracted result.
        instruction: A single instruction to process.
        return: A dictionary containing the processed result.
        """
        examples_fix = instruction["正确示例"].split('\t')
        examples_err = instruction['错误示例'].split('\t')
        output_instances = []
        
        for err, fix in zip(examples_err, examples_fix):
            matching_instances = self.extraction(fix)
            
            if matching_instances:
                for instance in matching_instances:
                    label, source_sentence, target_sentence = self.create_label_and_source(err, fix, instance)
                    output_instance = {
                        'domain': instruction['域'],
                        'instruction': instruction['指令'],
                        'input': source_sentence,
                        'output': target_sentence,
                        'typo': [err, fix],
                        'label': label,
                        'instruction_type': instruction['instruction_type'],
                        'instruction_index': instruction['instruction_index'],
                        'data_source': file
                    }
                    output_instances.append(output_instance)
        
            else:
                continue
        
        return output_instances

def main():
    parser = argparse.ArgumentParser()
    
    # Data config
    parser.add_argument("--data_dir", type=str, default="./dataset/",
                        help="Directory that contains all the data.")
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    instruction_data = processor.get_instruction_data(args.data_dir, 'instructions.json')
    instruction_domains = processor.get_domains_from_instruction([instruction['域'] for instruction in instruction_data])
    extracted_data = []
    
    with open(args.data_dir + 'instructions_extracted.jsonl', "w", encoding="utf-8") as extracted_file:
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if "general" in root:  # Check if the "general" directory is part of the root path
                    source_data = processor.get_source_data(root,"", file)
                    extractor = Extract(source_data)
                    
                    for instruction in tqdm(instruction_data, total=len(instruction_data)):
                        output_instances = extractor.extract_from_source(instruction, file)
                        if output_instances:
                            try:
                                extracted_data.extend(output_instances)
                                extracted_file.write("\n".join(json.dumps(output_instance, ensure_ascii=False) for output_instance in output_instances) + "\n")
                            except (TypeError, IOError) as e:
                                # Handle exceptions such as serialization errors or file write errors
                                logger.error(f"Error writing to file: {e}")
                        
                        
                elif "domain_specific" in root:
                    """
                    Print the domain of the source data.
                    Print the list of domains in the instructions.
                    User chooses the domain that matches the current source data.
                    And then do the extraction.
                    """
                    matching_domains = processor.user_domain_selection(file, instruction_domains)                    
                    source_data = processor.get_source_data(root,"", file)
                    
                    extractor = Extract(source_data)
                    
                    for instruction in tqdm(instruction_data, total=len(instruction_data)):
                        if instruction['域'] in [instruction_domains[i] for i in matching_domains]:
                            output_instances = extractor.extract_from_source(instruction, file)
                            if output_instances:
                                try:
                                    extracted_data.extend(output_instances)
                                    extracted_file.write("\n".join(json.dumps(output_instance, ensure_ascii=False) for output_instance in output_instances) + "\n")
                                except (TypeError, IOError) as e:
                                    # Handle exceptions such as serialization errors or file write errors
                                    logger.error(f"Error writing to file: {e}")
                logger.info(f"Processed file: {file}")

        
    with open(args.data_dir + 'instructions_non_extracted.jsonl', "w", encoding="utf-8") as non_extracted_file:
        for instruction in tqdm(instruction_data, total=len(instruction_data)):
            examples_fix = instruction["正确示例"].split('\t')
            examples_err = instruction['错误示例'].split('\t')
            
            for err, fix in zip(examples_err, examples_fix):
                label = 1 if err != fix else 0
                output_instance = {
                    'domain': instruction['域'],
                    'instruction': instruction['指令'],
                    'input': None,
                    'output': None,
                    'typo': [err, fix],
                    'label': label,
                    'instruction_type': instruction['instruction_type'],
                    'instruction_index': instruction['instruction_index'],
                    'data_source': None
                }
                for item in extracted_data:
                    if item.get('instruction_index') == instruction['instruction_index'] and item.get('typo') == output_instance['typo']:
                        break
                else:
                    # This will be executed if no `break` was encountered in the loop
                    non_extracted_file.write(json.dumps(output_instance, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    main()