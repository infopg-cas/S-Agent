import json
import pprint
import sys
from src.utils import ROOT_DIR

import os
folder = ''
prompt_file = '/data/hotpot/hotpot_train_v1_simplified.json'
print(ROOT_DIR + prompt_file)

with open(ROOT_DIR + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

pprint.pprint(prompt_dict)
print(len(prompt_dict))
