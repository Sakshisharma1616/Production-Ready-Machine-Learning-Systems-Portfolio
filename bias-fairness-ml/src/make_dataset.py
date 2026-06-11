from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Paths, DataConfig, TrainConfig
from utils import ensure_dir, save_json

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize string columns
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
    # Replace common missing markers
    df = df.replace({"?": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df

def split_data(
    df: pd.DataFrame, target_col: str, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    return train_df, test_df

def main(raw_path: Path, train_out: Path, test_out: Path) -> None:
    paths = Paths()
    dcfg = DataConfig()
    tcfg = TrainConfig()

    ensure_dir(paths.data_dir)

    df = pd.read_csv(raw_path)
    df = basic_cleaning(df)

    # Drop rows missing target or sensitive cols (simple, reproducible baseline)
    needed = [dcfg.target_col, *dcfg.sensitive_cols]
    df = df.dropna(subset=needed)

    train_df, test_df = split_data(df, dcfg.target_col, tcfg.test_size, tcfg.random_state)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    meta = {
        "data_config": asdict(dcfg),
        "train_config": asdict(tcfg),
        "raw_shape": df.shape,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "dropped_na_on": needed,
    }
    save_json(meta, paths.reports_dir / "dataset_meta.json")

    print(f"[OK] Train saved: {train_out} | shape={train_df.shape}")
    print(f"[OK] Test  saved: {test_out}  | shape={test_df.shape}")
    print(f"[OK] Meta  saved: {paths.reports_dir / 'dataset_meta.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default=str(Paths().data_dir / "raw_adult.csv"))
    parser.add_argument("--train_out", type=str, default=str(Paths().data_dir / "train.csv"))
    parser.add_argument("--test_out", type=str, default=str(Paths().data_dir / "test.csv"))
    args = parser.parse_args()
    main(Path(args.raw), Path(args.train_out), Path(args.test_out))
