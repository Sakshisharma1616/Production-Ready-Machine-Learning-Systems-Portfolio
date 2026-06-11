import argparse
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb

from src.ranking.data.negative_sampling import make_ranking_dataset
from src.ranking.data.splits import time_split, random_split
from src.ranking.features.user_features import add_user_aggregate_features
from src.ranking.features.item_features import add_item_aggregate_features
from src.ranking.features.context_features import add_context_features
from src.ranking.models.evaluate import evaluate_ranking
from src.ranking.models.registry import save_model, RegistryPaths

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _one_hot_encode(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cat_cols, dummy_na=True)

def _build_group_sizes(df: pd.DataFrame) -> np.ndarray:
    # group sizes for LightGBM ranker (must align with row order)
    grouped = df.groupby(["user_id", "session_id"], sort=False).size().to_numpy()
    return grouped.astype(int)

def main(config_path: str) -> None:
    cfg = _load_yaml(config_path)
    seed = int(cfg["project"]["seed"])
    np.random.seed(seed)

    raw_dir = cfg["paths"]["raw_dir"]
    processed_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["artifacts_models"]
    reports_dir = cfg["paths"]["artifacts_reports"]

    _ensure_dir(processed_dir)
    _ensure_dir(models_dir)
    _ensure_dir(reports_dir)

    # Load raw
    users = pd.read_csv(os.path.join(raw_dir, "users.csv"))
    items = pd.read_csv(os.path.join(raw_dir, "items.csv"))
    interactions = pd.read_csv(os.path.join(raw_dir, "interactions.csv"), parse_dates=["timestamp"])

    # Build ranking dataset (neg sampling)
    ds = make_ranking_dataset(
        interactions=interactions,
        users=users,
        items=items,
        negatives_per_positive=int(cfg["negative_sampling"]["negatives_per_positive"]),
        strategy=str(cfg["negative_sampling"]["sampling_strategy"]),
        seed=seed,
    )

    # Feature engineering (offline feature-store style)
    ds = add_user_aggregate_features(ds, interactions, window_days=int(cfg["features"]["history_window_days"]))
    ds = add_item_aggregate_features(ds, interactions, window_days=int(cfg["features"]["history_window_days"]))
    ds = add_context_features(ds)

    # Additional cross features (cheap but effective)
    ds["item_age"] = datetime.now().year - ds["release_year"].astype(int)
    ds["is_kids_content"] = (ds["genre"] == "Kids").astype(int)
    ds["kids_mismatch"] = ((ds["is_kids_profile"].astype(int) == 1) & (ds["genre"] != "Kids")).astype(int)

    # Split
    split_cfg = cfg["splits"]
    if split_cfg["strategy"] == "time":
        train_df, val_df, test_df = time_split(ds, float(split_cfg["train_frac"]), float(split_cfg["val_frac"]), float(split_cfg["test_frac"]))
    else:
        train_df, val_df, test_df = random_split(ds, float(split_cfg["train_frac"]), float(split_cfg["val_frac"]), float(split_cfg["test_frac"]), seed=seed)

    # Persist processed
    train_path = os.path.join(processed_dir, "train.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print("✅ Saved processed splits to data/processed")

    # Prepare model inputs
    target = "label"
    group_cols = ["user_id", "session_id"]

    drop_cols = ["timestamp", "watch_minutes"]  # label leakage-ish & not always available online
    feature_df_train = train_df.drop(columns=[target] + drop_cols)
    feature_df_val = val_df.drop(columns=[target] + drop_cols)
    feature_df_test = test_df.drop(columns=[target] + drop_cols)

    # Categorical columns
    cat_cols = ["user_id", "session_id", "item_id", "age_bucket", "country", "genre", "maturity", "device"]
    # We keep group keys in dataframes for grouping, but remove IDs from features after we compute group sizes.
    # For simplicity: one-hot everything including IDs (works for demo; production would use embeddings or hashing)
    X_train = _one_hot_encode(feature_df_train, cat_cols)
    X_val = _one_hot_encode(feature_df_val, cat_cols)
    X_test = _one_hot_encode(feature_df_test, cat_cols)

    # Align columns across splits
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[target].astype(int).to_numpy()
    y_val = val_df[target].astype(int).to_numpy()

    # Group sizes (must follow the order of rows)
    train_group = _build_group_sizes(train_df[group_cols])
    val_group = _build_group_sizes(val_df[group_cols])

    # LightGBM ranker
    model_cfg = cfg["model"]
    train_set = lgb.Dataset(X_train, label=y_train, group=train_group, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, group=val_group, reference=train_set, free_raw_data=False)

    params = {
        "objective": model_cfg["objective"],
        "metric": model_cfg["metric"],
        "boosting_type": model_cfg["boosting_type"],
        "num_leaves": int(model_cfg["num_leaves"]),
        "learning_rate": float(model_cfg["learning_rate"]),
        "min_data_in_leaf": int(model_cfg["min_data_in_leaf"]),
        "feature_fraction": float(model_cfg["feature_fraction"]),
        "bagging_fraction": float(model_cfg["bagging_fraction"]),
        "bagging_freq": int(model_cfg["bagging_freq"]),
        "lambda_l1": float(model_cfg["lambda_l1"]),
        "lambda_l2": float(model_cfg["lambda_l2"]),
        "seed": int(model_cfg["random_state"]),
        "verbosity": -1,
    }

    num_boost_round = int(model_cfg["n_estimators"])
    early_stopping_rounds = int(cfg["training"]["early_stopping_rounds"])

    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    # Evaluate on val/test using our metrics
    val_scores = booster.predict(X_val, num_iteration=booster.best_iteration)
    test_scores = booster.predict(X_test, num_iteration=booster.best_iteration)

    val_eval_df = val_df[["user_id", "session_id", "item_id", "label"]].copy()
    val_eval_df["score"] = val_scores

    test_eval_df = test_df[["user_id", "session_id", "item_id", "label"]].copy()
    test_eval_df["score"] = test_scores

    k = int(cfg["features"]["eval_k"])
    val_metrics = evaluate_ranking(val_eval_df, score_col="score", k=k)
    test_metrics = evaluate_ranking(test_eval_df, score_col="score", k=k)

    metrics = {"val": val_metrics, "test": test_metrics, "k": k, "best_iteration": int(booster.best_iteration)}
    with open(os.path.join(reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("📈 Metrics:", json.dumps(metrics, indent=2))

    meta = {
        "model_type": "LightGBM LambdaRank",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "best_iteration": int(booster.best_iteration),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "features": list(X_train.columns),
        "label_definition": "0=no-engagement negative, 1=click, 2=short-play, 3=long-play",
        "metrics": metrics,
    }

    save_model(booster, meta, RegistryPaths(models_dir=models_dir))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
