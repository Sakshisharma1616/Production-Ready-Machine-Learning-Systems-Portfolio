from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from config import Paths, DataConfig, TrainConfig
from utils import ensure_dir

def build_pipeline(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=TrainConfig().max_iter)

    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])

def infer_columns(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target_col])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    return cat_cols, num_cols

def main(train_path: Path, model_out: Path) -> None:
    paths = Paths()
    cfg = DataConfig()
    tcfg = TrainConfig()

    ensure_dir(paths.models_dir)

    df = pd.read_csv(train_path)

    # Convert target to binary 0/1 (Adult usually has values like '>50K' and '<=50K')
    y_raw = df[cfg.target_col].astype(str)
    y = (y_raw.str.contains(">50K") | y_raw.str.contains("1")).astype(int)

    X = df.drop(columns=[cfg.target_col])

    cat_cols, num_cols = infer_columns(df, cfg.target_col)
    pipe = build_pipeline(cat_cols=cat_cols, num_cols=num_cols)

    pipe.fit(X, y)

    joblib.dump(
        {"model": pipe, "cat_cols": cat_cols, "num_cols": num_cols, "random_state": tcfg.random_state},
        model_out,
    )

    print(f"[OK] Trained baseline model")
    print(f"[OK] Saved model: {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=str(Paths().data_dir / "train.csv"))
    parser.add_argument("--model_out", type=str, default=str(Paths().models_dir / "baseline.joblib"))
    args = parser.parse_args()
    main(Path(args.train), Path(args.model_out))
