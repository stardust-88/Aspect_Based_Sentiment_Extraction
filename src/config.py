# configuration class for hyperparameters and other configurations

import torch
import os

class Config:

      def __init__(self):

        # general hyperparameters
        self.MAX_LEN = 250
        self.BATCH_SIZE = 16
        self.EPOCHS = 20
        self.learning_rate = 0.00001
        
        # train, val and test split ratios (when using train_test_split of scikit learn)
        self.TRAIN_VAL_SPLIT = 0.3
        self.VAL_TEST_SPLIT = 0.5

        # paths
        self.path = os.getcwd()[:-4]
        self.data_path = f'{self.path}/data'
        self.train_file_path = f'{self.path}/data/train.csv'
        self.test_file_path = f'{self.path}/data/results/test.csv'
        self.model_save_path = f'{self.path}/models'
        self.results_save_path = f'{self.path}/data/results'

        # model types
        self.bert_base_model = True # if want to use the BertModel, otherwise DistilBertModel
        self.case_based_model = True # if want to use the cased or uncased version

        # others
        self.num_workers = 0
        self.RANDOM_SEED = 43
        self.train_shuffle = True
        self.valid_shuffle = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")