import pandas as pd
from sklearn import model_selection
from config import Config

conf = Config()

def define_folds(df):
    
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.label.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
        
    df.to_csv(f'{conf.data_path}/train_folds.csv', index=False)
        
    return df
