import torch
from torch.utils.data import Dataset, DataLoader
from model import ABSA_Model
from config import Config
from utils import return_tokenizer, basic_preprocessing
import numpy as np

conf = Config()
#device = conf.device
device = torch.device('cpu')

model = ABSA_Model()
model.load_state_dict(torch.load(f'{conf.model_save_path}/best_model_state.bin'))
model = model.to(device)

class_names = ['Negative', 'Neutral', 'Positive']

# building up the custom dataset class for our test file
class Test_ABSA_Dataset(Dataset):
    def __init__(self, texts, aspects, tokenizer, max_len):
        self.texts = texts
        self.aspects = aspects
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, index):
        text = str(self.texts[index])
        aspect = str(self.aspects[index])
        
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
        }


# dataloader function for test file
def create_dataloader(df, tokenizer, max_len, batch_size, shuffle=False):
    
    dataset = Test_ABSA_Dataset(
        texts = df.cleaned_text.to_numpy(),
        aspects = df.cleaned_aspect.to_numpy(),
        tokenizer = tokenizer,
        max_len = max_len
      )

    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = conf.num_workers, shuffle = shuffle)
    return dataloader


def generate_predictions(file_size, dataloader):
    
    predictions = []
    dataloader_iterator = iter(dataloader)
    
    model.eval()
    for i in range(file_size):
        data = next(dataloader_iterator)
        
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        
        with torch.no_grad():
            output = model(input_ids, attention_mask)
        
        _, prediction = torch.max(output, dim=1)
        predictions.append(prediction.cpu())
        
    return predictions
              
    
if __name__ == '__main__':
    
    # read the test file and do the basic preprocessing
    test_df = pd.read_csv(conf.test_file_path)
    test_df['cleaned_text']=test_df['text'].apply(lambda text:basic_preprocessing(text))
    test_df['cleaned_aspect']=test_df['aspect'].apply(lambda aspect:basic_preprocessing(aspect))
    
    # use the the tokenizer that was used for the training, see config file for info.
    tokenizer = return_tokenizer()
    
    # create the dataloader for the test file
    dataloader = create_dataloader(test_df, tokenizer, conf.MAX_LEN, 1)
    
    # get the list of predictions for all the entries in the test file
    predictions = generate_predictions(test_df.shape[0], dataloader)
    test_df['label'] = np.array(predictions)
    
    # save the test file with predictions
    test_df.to_csv(f'{conf.results_save_path}/submission_for_test.csv', index = False)
    