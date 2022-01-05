import torch
from model import ABSA_Model
from config import Config
from utils import return_tokenizer, basic_preprocessing

conf = Config()
device = conf.device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

class_names = ['Negative', 'Neutral', 'Positive']

class Model:
    def __init__(self):
        
        self.tokenizer = return_tokenizer()
        absa_model = ABSA_Model()
        absa_model.load_state_dict(torch.load(f'{conf.model_save_path}/best_model_state.bin'))
        absa_model = absa_model.to(conf.device)
        self.absa_model = absa_model.eval()
        
    def predict(self, text, aspect):
        
        text = basic_preprocessing(text)
        aspect = basic_preprocessing(aspect)
        
        encoded_text = self.tokenizer(
            text,
            aspect,
            add_special_tokens = True,
            max_length = conf.MAX_LEN,
            return_token_type_ids = False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        with torch.no_grad():
            output = self.absa_model(input_ids, attention_mask)
        
        _, prediction = torch.max(output, dim=1)
        #prediction = prediction.cpu().numpy()
        
        return class_names[prediction]  

_model = Model() 

def get_model():
    return _model
    
# prediction on custom input----------------------------------------------------------------------------------
def prediction_on_custom_input():
    
    while(True):
        print()
        text = input("Enter the sentence/text: ")
        print()
        aspect = input("Enter the aspect word/phrase from the above sentence: ")
        print()
    
        sentiment = _model.predict(text, aspect)
        print(f"Sentiment expressed towards the aspect in the text:  {sentiment}")
        
        print()
        ok = input("Want to test on more examples (yes/no): ")
        if ok=="yes" or ok=="y":
            continue
        else:
            break
#-------------------------------------------------------------------------------------------------------------

# prediction on example texts---------------------------------------------------------------------------------
def prediction_on_example():
    text = "I like pizza very much"
    aspect = "pizza"
    print(f"text: {text}")
    print(f"aspect/phrase: {aspect}")
    print()
    sentiment = _model.predict(text, aspect)
    print(f"Sentiment expressed towards the aspect in the text:  {sentiment}")
#-------------------------------------------------------------------------------------------------------------   
      
    
def for_predictions():
    
    print()
    ans = input("See the prediction on provided example? (yes/no): ")
    if ans == "yes" or ans == "y":
         prediction_on_example()
    else:
        ans = input("Want to see predictions on custom inputs? (yes/no): ")
        if ans == "yes" or ans=="y":
            prediction_on_custom_input()
        else:
            return
        
    return

# uncomment this to test on inputs
for_predictions()