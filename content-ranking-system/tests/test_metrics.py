import numpy as np
import pandas as pd
from src.ranking.models.evaluate import ndcg_at_k, map_at_k, evaluate_ranking

def test_ndcg_basic():
    rels = np.array([3, 2, 0, 1])
    assert 0.0 <= ndcg_at_k(rels, 3) <= 1.0

def test_map_basic():
    rels = np.array([1, 0, 1, 0])
    assert 0.0 <= map_at_k(rels, 4) <= 1.0

def test_group_eval_runs():
    df = pd.DataFrame({
        "user_id": ["u1","u1","u1","u2","u2"],
        "session_id": ["s1","s1","s1","s2","s2"],
        "label": [3,0,1,2,0],
        "score": [0.9,0.1,0.3,0.8,0.2]
    })
    out = evaluate_ranking(df, "score", k=3)
    assert "NDCG@3" in out and "MAP@3" in out
