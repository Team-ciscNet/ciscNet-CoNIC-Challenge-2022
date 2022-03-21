import os

import pandas as pd
import joblib


splits = [{
        'train': pd.read_csv(
            os.path.join('exp_output', 'local', 'data', 'ids_train_80.csv')
        )['ids'].values,
        'valid': pd.read_csv(
            os.path.join('exp_output', 'local', 'data', 'ids_val_80.csv')
        )['ids'].values,
}]

joblib.dump(splits, 'splits.dat')
