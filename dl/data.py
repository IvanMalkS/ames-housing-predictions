import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def add_features(df):
    df = df.copy()
    df['HouseAge']   = df['YrSold'] - df['YearBuilt']
    df['RemodAge']   = df['YrSold'] - df['YearRemodAdd']
    df['TotalSF']    = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBaths'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['PorchArea']  = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['QualArea']   = df['OverallQual'] * df['GrLivArea']
    return df


class AmesDataset(Dataset):
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y     = torch.tensor(y,     dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'num':   self.X_num[idx],
            'cat':   self.X_cat[idx],
            'label': self.y[idx],
        }


def _build_pipes(X_train: pd.DataFrame, num_cols: list, cat_cols: list):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    num_pipe.fit(X_train[num_cols])
    cat_pipe.fit(X_train[cat_cols])
    return num_pipe, cat_pipe


def _transform(X: pd.DataFrame, num_pipe, cat_pipe, num_cols: list, cat_cols: list):
    X_num = num_pipe.transform(X[num_cols])
    X_cat = cat_pipe.transform(X[cat_cols]).astype(int) + 1
    return X_num, X_cat


def download_data(csv_path: str) -> None:
    import ssl

    print(f'Downloading Ames Housing dataset to {csv_path} ...')
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        dataset = fetch_openml('house_prices', version=1, as_frame=True, parser='auto')
    finally:
        ssl._create_default_https_context = _orig

    df = dataset.frame
    if 'Id' not in df.columns:
        df.insert(0, 'Id', range(1, len(df) + 1))
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f'Saved {len(df)} rows to {csv_path}')


def prepare_data(config, train_csv_path: str):
    np.random.seed(config.general.seed)

    if not os.path.exists(train_csv_path):
        download_data(train_csv_path)

    df = pd.read_csv(train_csv_path)
    df = df.drop('Id', axis=1, errors='ignore')
    df = df.drop(df[df['GrLivArea'] > 4000].index)
    df['SalePrice'] = np.log1p(df['SalePrice'])
    df = add_features(df)

    X = df.drop('SalePrice', axis=1).reset_index(drop=True)
    y = df['SalePrice'].values

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    return X, y, num_cols, cat_cols


def get_fold_loaders(config, X: pd.DataFrame, y: np.ndarray,
                     train_idx: np.ndarray, val_idx: np.ndarray,
                     num_cols: list, cat_cols: list):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    num_pipe, cat_pipe = _build_pipes(X_train, num_cols, cat_cols)
    X_num_train, X_cat_train = _transform(X_train, num_pipe, cat_pipe, num_cols, cat_cols)
    X_num_val,   X_cat_val   = _transform(X_val,   num_pipe, cat_pipe, num_cols, cat_cols)

    cat_sizes = [int(X_cat_train[:, i].max()) + 1 for i in range(X_cat_train.shape[1])]

    dl_params = dict(batch_size=config.dataloader.batch_size, num_workers=config.dataloader.num_workers)
    train_loader = DataLoader(
        AmesDataset(X_num_train, X_cat_train, y_train),
        shuffle=config.dataloader.shuffle,
        drop_last=config.dataloader.drop_last,
        **dl_params,
    )
    val_loader = DataLoader(
        AmesDataset(X_num_val, X_cat_val, y_val),
        shuffle=False,
        **dl_params,
    )

    meta = {
        'num_features': X_num_train.shape[1],
        'cat_sizes':    cat_sizes,
        'num_cols':     num_cols,
        'cat_cols':     cat_cols,
        'num_pipe':     num_pipe,
        'cat_pipe':     cat_pipe,
    }
    return train_loader, val_loader, meta
