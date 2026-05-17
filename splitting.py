import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def split_data(y: np.ndarray, df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = []
    
    for train_val_idx, test_idx in skf.split(np.zeros(len(y)), y):
        y_train_val = y[train_val_idx]
        
        skf_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_sub_idx, val_sub_idx = next(skf_val.split(np.zeros(len(y_train_val)), y_train_val))
        
        idx_train = train_val_idx[train_sub_idx]
        idx_val = train_val_idx[val_sub_idx]
        
        splits.append((idx_train, idx_val, test_idx))
        
    return splits
  
