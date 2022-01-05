import transformers as tr
import torch
from torch.utils.data import Dataset, DataLoader
import re
import os
import numpy as np
import random
from config import Config

conf = Config()

# building up the custom dataset class for our datset
class ABSA_Dataset(Dataset):
    def __init__(self, texts, aspects, targets, tokenizer, max_len):
        self.texts = texts
        self.aspects = aspects
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, index):
        text = str(self.texts[index])
        aspect = str(self.aspects[index])
        target = self.targets[index]
        
        encoding = self.tokenizer(
          text,
          aspect,
          add_special_tokens = True,
          max_length = self.max_len,
          return_token_type_ids = False,
          padding = 'max_length',
          truncation = True,
          return_attention_mask = True,
          return_tensors = 'pt',
        )

        return {
          'text': text,
          'aspect': aspect,
          'input_ids': encoding['input_ids'].flatten(),  # from torch.Size([1,max_len]) to torch.Size([max_len])
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }
    
# tokenizer for a particular bert model
def return_tokenizer():
    
    if conf.bert_base_model:
        if conf.case_based_model:
            tokenizer = tr.BertTokenizer.from_pretrained('bert-base-cased')
        else:
             tokenizer = tr.BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        if conf.case_based_model:
            tokenizer = tr.DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        else:
            tokenizer = tr.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
    return tokenizer

# function for creating dataloaders for train and test datasets
def create_dataloader(df, tokenizer, max_len, batch_size, shuffle=False):
    
    dataset = ABSA_Dataset(
        texts = df.cleaned_text.to_numpy(),
        aspects = df.cleaned_aspect.to_numpy(),
        targets = df.label.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
      )

    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = conf.num_workers, shuffle = shuffle)
    return dataloader

# utility for basic preprocessing of texts
def basic_preprocessing(doc):
    
    # removing the URLs
    url_pattern = r'\S*https?:\S*'
    data = re.sub(url_pattern,'',doc)

    # removing the usernames:
    username_pattern = r'@[^\s]+'
    data = re.sub(username_pattern,'',data)

    # remove special characters 
    special_chars = r'[^\w\s]|_'
    data = re.sub(special_chars, '', data)

    # remving patterns in square brackets
    sq_bracks_pat = r'\[.*?\]'
    data = re.sub(sq_bracks_pat, '', data)

    # removing unwanted html tags
    html_pattern = r'<.*?>+'
    data = re.sub(html_pattern, '', data)

    return data

# setting random seeds-----------------------------------------
def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
