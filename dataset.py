import os 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

class PromptDataset(Dataset):
    def __init__(self,
                path=None,
                name=None,
                tokenizer=None,
                temps=None,
                max_length = 128):
        super().__init__()
        self.max_length = max_length
        self.temps = temps
        self.get_labels(tokenizer)
        f_out = open(path + "/" + name + '.txt', "r")
        self.lines = f_out.readlines()
        f_out.close()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        txt, label = line.split('\t')
        sent_ids = self.tokenizer.encode(txt, add_special_tokens = False)
        prompt = self.temp_ids[label]['mask_ids'][0]           #numpy array
        lm_label = self.temp_ids[label]['label_ids'][0]         #int
        input_ids = sent_ids + prompt
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2] + [1] * (self.max_length - length))
        attention_mask = (input_ids != 1).long()
        lm_label = torch.LongTensor([lm_label])
        loc = length - 2
        return input_ids, attention_mask, lm_label, loc

    def get_labels(self, tokenizer):
        #total = {}
        self.temp_ids = {}
        for name in self.temps:
            #last = 0
            self.temp_ids[name] = {}
            self.temp_ids[name]['label_ids'] = []
            self.temp_ids[name]['mask_ids'] = []
            temp = self.temps[name]['temp']                                  
            _temp = temp.copy()
            _label = self.temps[name]['label']

            for i in range(len(_temp)):
                if _temp[i] == tokenizer.mask_token:
                    _temp[i] = _label
                    _label_index = i
            
            original = tokenizer.encode(' '.join(temp), add_special_tokens = False)
            final = tokenizer.encode(' '.join(_temp), add_special_tokens = False)

            assert len(original) == len(final)
            self.temp_ids[name]['label_ids'].append(final[_label_index])

            self.temp_ids[name]['mask_ids'].append(original)
        print(self.temp_ids)
