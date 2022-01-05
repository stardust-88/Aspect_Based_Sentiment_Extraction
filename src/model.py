import torch
import torch.nn as nn
import transformers as tr
from config import Config

conf = Config()

class ABSA_Model(nn.Module):
    
    def __init__(self, dropout = 0.5):
        super(ABSA_Model, self).__init__()

        if conf.bert_base_model:
            if conf.case_based_model:
                self.bert_model = tr.BertModel.from_pretrained('bert-base-cased')
            else:
                self.bert_model = tr.BertModel.from_pretrained('bert-base-uncased')
        else:
            if conf.case_based_model:
                self.bert_model = tr.DistilBertModel.from_pretrained('distilbert-base-cased')
            else:
                self.bert_model = tr.DistilBertModel.from_pretrained('distilbert-base-uncased')
                
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, mask):

        _, pooled_output = self.bert_model(input_ids = input_ids, attention_mask = mask, return_dict = False) # BertModel
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.softmax(x)

        return x