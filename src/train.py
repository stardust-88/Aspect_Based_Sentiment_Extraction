import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers as tr

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from create_folds import define_folds

import numpy as np
from collections import defaultdict
from collections import  Counter
import pandas as pd

from model import ABSA_Model
from config import Config
from utils import ABSA_Dataset, create_dataloader, return_tokenizer, basic_preprocessing, set_all_seeds

conf = Config()

def train_fn(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    
    model = model.train()
    losses = []
    correct_predictions = 0

    for data in data_loader:

        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        outputs = model(input_ids, attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)

        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_fn(model, data_loader, loss_fn, device, n_examples):
    
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(input_ids, attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def execute(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, scheduler, df_train_len, df_val_len):
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(conf.EPOCHS):

        print(f'Epoch {epoch + 1}/{conf.EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_fn(model, train_dataloader, loss_fn, optimizer, device, scheduler, df_train_len)

        print(f'Train loss: {train_loss}, Train Accuracy: {train_acc*100}')

        val_acc, val_loss = eval_fn(model, val_dataloader, loss_fn, device, df_val_len)

        print(f'Validation loss: {val_loss}, Validation Accuracy: {val_acc*100}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), f'{conf.model_save_path}/best_model_state.bin')
            best_accuracy = val_acc
        
    print('Execution Complete')
    print('-' * 10)

    return history

def regular_train_test_split(train_df):
    
    df_train, df_test = train_test_split(
      train_df,
      test_size=conf.TRAIN_VAL_SPLIT,
      random_state=conf.RANDOM_SEED
    )

    df_val, df_test = train_test_split(
      df_test,
      test_size=conf.VAL_TEST_SPLIT,
      random_state=conf.RANDOM_SEED
    )
    
    return df_train, df_val, df_test


def save_info(history):
    
    train_acc = []
    val_acc = []
    for acc in history['train_acc']:
        train_acc.append(acc.cpu().numpy())

    for acc in history['val_acc']:
        val_acc.append(acc.cpu().numpy())
        
    with open(f"{conf.results_save_path}/train_acc.txt", "wb") as fp:
        pickle.dump(train_acc, fp)

    with open(f"{conf.results_save_path}/val_acc.txt", "wb") as fp:
        pickle.dump(val_acc, fp)

    with open(f"{conf.results_save_path}/train_loss.txt", "wb") as fp:
        pickle.dump(history['train_loss'], fp)

    with open(f"{conf.results_save_path}/val_loss.txt", "wb") as fp:
        pickle.dump(history['val_loss'], fp)


if __name__ == '__main__':
    
    device = conf.device
    set_all_seeds(conf.RANDOM_SEED)
    
    # read the train csv file
    train_df = pd.read_csv(conf.train_file_path)
    
    # apply some basic preprocessing
    train_df['cleaned_text']=train_df['text'].apply(lambda text:basic_preprocessing(text))
    train_df['cleaned_aspect']=train_df['aspect'].apply(lambda aspect:basic_preprocessing(aspect))
    
    # splitting data using stratified kfold cross validation
    train_df = define_folds(train_df)
    fold = 0
    df_train = train_df[train_df['kfold'] != fold]
    df_valid = train_df[train_df['kfold'] == fold]
    
    # splitting the training data using regular train_test_split of scikit learn
    #df_train, df_val, df_test = regular_train_test_split(train_df)
    
    # get bert model tokenizer
    tokenizer = return_tokenizer()
    
    # setting up the train and test dataloaders
    train_dataloader = create_dataloader(df_train, tokenizer, conf.MAX_LEN, conf.BATCH_SIZE, conf.train_shuffle)
    val_dataloader = create_dataloader(df_valid, tokenizer, conf.MAX_LEN, conf.BATCH_SIZE, conf.train_shuffle)
    #test_dataloader = create_dataloader(df_test, tokenizer, conf.MAX_LEN, conf.BATCH_SIZE, conf.valid_shuffle)
    
    # initialize absa model
    model = ABSA_Model()
    model = model.to(device)
    
    # setting up the optimizers, schedulers and the loss function
    optimizer = tr.AdamW(model.parameters(), lr=conf.learning_rate, correct_bias=False)

    total_steps = len(train_dataloader) * conf.EPOCHS
    scheduler = tr.get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
    )

    loss_fn = nn.NLLLoss().to(device)
    
    # executing the model
    tr.logging.set_verbosity_error()
    history = execute(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, scheduler, len(df_train), len(df_valid))
    
    # saving the losses and accuracies
    save_info(history)
