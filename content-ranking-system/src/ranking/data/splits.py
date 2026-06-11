import pandas as pd
import numpy as np

def time_split(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test

def random_split(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int):
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = df.iloc[idx[:n_train]].copy()
    val = df.iloc[idx[n_train:n_train + n_val]].copy()
    test = df.iloc[idx[n_train + n_val:]].copy()
    return train, val, test

