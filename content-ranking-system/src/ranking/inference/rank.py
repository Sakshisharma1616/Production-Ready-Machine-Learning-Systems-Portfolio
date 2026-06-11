from __future__ import annotations
import os
import pandas as pd
import numpy as np

from src.ranking.models.registry import load_model, RegistryPaths

def _one_hot_from_meta(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    x = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    x = x.reindex(columns=feature_cols, fill_value=0)
    return x

def rank_candidates(
    user_row: dict,
    item_rows: list[dict],
    context: dict,
    models_dir: str = "artifacts/models",
) -> list[dict]:
    """
    Online ranking:
    - Build feature rows for each candidate item
    - One-hot to match training columns (from model_meta)
    - Score with LightGBM
    """
    model, meta = load_model(RegistryPaths(models_dir=models_dir))
    feature_cols = meta["features"]

    # Build dataframe for candidates
    rows = []
    for it in item_rows:
        row = {}
        row.update(user_row)
        row.update(it)
        row.update({
            "device": context.get("device", "tv"),
            "hour": int(context.get("hour", 20)),
            "day_of_week": int(context.get("day_of_week", 2)),
            "is_prime_time": int(1 if 19 <= int(context.get("hour", 20)) <= 23 else 0),
            "is_weekend": int(1 if int(context.get("day_of_week", 2)) in [5, 6] else 0),
        })
        # derived
        row["item_age"] = int(context.get("current_year", 2026)) - int(row.get("release_year", 2020))
        row["is_kids_content"] = int(1 if row.get("genre") == "Kids" else 0)
        row["kids_mismatch"] = int(1 if int(row.get("is_kids_profile", 0)) == 1 and row.get("genre") != "Kids" else 0)

        # placeholders for offline aggregates if not available online
        for c in ["u_watch_mins_30d", "u_plays_30d", "u_clicks_30d", "u_play_rate_30d",
                  "i_watch_mins_30d", "i_plays_30d", "i_clicks_30d", "i_play_rate_30d"]:
            row.setdefault(c, 0.0)
        # group keys required by encoding schema if included
        row.setdefault("session_id", context.get("session_id", "s_online"))
        rows.append(row)

    df = pd.DataFrame(rows)

    cat_cols = ["user_id", "session_id", "item_id", "age_bucket", "country", "genre", "maturity", "device"]
    X = _one_hot_from_meta(df, feature_cols=feature_cols, cat_cols=cat_cols)

    scores = model.predict(X, num_iteration=meta.get("best_iteration", None))
    df["score"] = scores

    out = df[["user_id", "item_id", "score"]].sort_values("score", ascending=False)
    return out.to_dict(orient="records")
