import os.path as osp
from typing import Union
import json
import logging
import random
import ipdb

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Prompter(object):
    __slots__ = ("template", "_verbose", "shot_bank", "test_mode", "_grouped_shots")
    
    def __init__(self, template_name: str , testset_dir: str, test_mode, verbose: bool = False):
        self._verbose = verbose
        self.test_mode = test_mode
       
        # Get Template
        if not template_name:
            ValueError("Template name must be given.")
        
        file_name = osp.join("./utils/templates/", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read the template \"{file_name}\"")
        with open(file_name, 'r', encoding='utf-8') as fp:
            self.template = json.load(fp)

        # Verbose
        if self._verbose:
            description = self.template.get('description', 'No description provided.')
            print(f"Using prompt template {template_name}: {description}")

        with open(testset_dir, "r", encoding="utf-8") as fp:
            self.shot_bank = json.load(fp)
        
        #for few shot examples
        if self.test_mode == "false_case_generation":
            # Filter the items with the different keywords after correction for noised few shots.
            self.shot_bank = [item for item in self.shot_bank if item['keyword_label'] == 0]
        
        self._grouped_shots = {}
        for item in self.shot_bank:
            domain = item.get('domain')
            if domain is None:
                raise ValueError(f"Data item missing 'domain' key: {item}")
            if domain not in self._grouped_shots:
                self._grouped_shots[domain] = []
            self._grouped_shots[domain].append(item)
            

        if self._verbose:
            print(f"Data successfully grouped into domains.")
      
    def get_few_shot_examples(self, case: dict, shots: int=3) -> list[dict[str, any]]:
        
        few_shot_examples = []
        # Look for shots with the same domain
        matching_items = []
        # Look for shots within the same domain
        if case['domain'] in self._grouped_shots:
            matching_items = self._grouped_shots[case['domain']]      
            
            # only fewshot the examples with the different keyword.
            matching_items = [
                item for item in matching_items
                if item['keyword'] != case['keyword']
            ]
        
            few_shot_examples.extend(random.sample(matching_items, min(len(matching_items), shots)))


        # Look for shots with the same keyword label
        if len(matching_items) < shots:
            logger.info(f"{len(matching_items)} examples found for domain '{case['domain']}', but {shots} are required. Getting {shots-len(matching_items)} examples from other domains.")
            
            if self.test_mode == "false_case_generation":
                few_shot_examples.extend(random.sample(self.shot_bank, shots - len(matching_items)))
            else:
                few_shot_examples.extend(random.sample(
                    [item for item in self.shot_bank if item['keyword_label'] == case['keyword_label']], shots - len(matching_items)
                ))
            

            
        return few_shot_examples
    
    def user_prompter(self, item: dict[str, any], test_mode: str) -> str:
        if "contrastive" in test_mode:
            contrastive_example = self.template['contrastive_example'].format(
                false_output = item['false_output']
            )
            # Do not provide false example for the current sample
            if item['false_output'] is None:
                contrastive_example = ""
            else:
                contrastive_example = self.template['contrastive_example'].format(
                    false_output = item['false_output']
                )
        else:
            contrastive_example = None
            
        return self.template[f'user_content_{test_mode}'].format(
            domain=item['domain'],
            instruction=item['instruction'],
            input=item['input'],
            err_keyword=item['keyword'][0],
            contrastive_example=contrastive_example
        )
    
    def assistant_prompter(self, item: dict[str, any], test_mode: str) -> str:
        return self.template[f'assistant_content_{test_mode}'].format(
            output=item['output'],
            corr_keyword=item['keyword'][1]
        )
    
    def gen_messages(self, item:dict[str, any], incontext_learning: int) -> str:
        messages = []
        few_shots = self.get_few_shot_examples(item, shots=incontext_learning)
        
        if self.test_mode != "false_case_generation":
            messages.append({"role": "system", "content": self.template['system_content']})
        

        for shot in few_shots:
            messages.append({"role": "user", "content": self.user_prompter(shot, self.test_mode)})
            messages.append({"role": "assistant", "content": self.assistant_prompter(shot, self.test_mode)})
            
        # Do not provide false case for the current sample.
        item['false_output'] = None
        messages.append({"role": "user", "content": self.user_prompter(item, self.test_mode)})
        return messages