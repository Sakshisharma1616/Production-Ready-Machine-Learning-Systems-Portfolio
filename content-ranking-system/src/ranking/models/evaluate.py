from __future__ import annotations
import numpy as np
import pandas as pd

def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.size == 0:
        return 0.0
    gains = (2 ** rels - 1)
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(gains / discounts))

def ndcg_at_k(rels: np.ndarray, k: int) -> float:
    dcg = dcg_at_k(rels, k)
    ideal = np.sort(rels)[::-1]
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg

def average_precision_at_k(binary_rels: np.ndarray, k: int) -> float:
    binary_rels = binary_rels[:k]
    if binary_rels.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, r in enumerate(binary_rels, start=1):
        if r == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0

def map_at_k(rels: np.ndarray, k: int, positive_threshold: int = 1) -> float:
    binary = (rels >= positive_threshold).astype(int)
    return average_precision_at_k(binary, k)

def evaluate_ranking(df: pd.DataFrame, score_col: str, k: int = 10) -> dict:
    """
    df must include: user_id, session_id, label, score_col
    """
    group_cols = ["user_id", "session_id"]
    ndcgs = []
    maps = []

    for _, g in df.groupby(group_cols):
        g = g.sort_values(score_col, ascending=False)
        rels = g["label"].to_numpy(dtype=int)
        ndcgs.append(ndcg_at_k(rels, k))
        maps.append(map_at_k(rels, k, positive_threshold=1))

    return {
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"MAP@{k}": float(np.mean(maps)) if maps else 0.0,
        "num_groups": int(df.groupby(group_cols).ngroups),
    }
