from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, f1_score
from config import Paths, DataConfig
from utils import ensure_dir, save_json

def group_threshold_search(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Choose per-group thresholds to reduce demographic parity difference
    while keeping accuracy reasonable.

    Simple heuristic:
    - For each group, try thresholds
    - Evaluate global DP diff on combined predictions
    - Select thresholds minimizing DP diff; break ties by higher accuracy.
    """
    groups = pd.Series(sensitive).astype(str).fillna("NA").unique().tolist()
    best = None

    # To keep search manageable: Cartesian product over groups could blow up.
    # We'll do greedy coordinate descent: iterate groups and optimize one at a time.
    current = {g: 0.5 for g in groups}

    def predict_with(th_map: Dict[str, float]) -> np.ndarray:
        yp = np.zeros_like(y_prob, dtype=int)
        for g in groups:
            mask = (pd.Series(sensitive).astype(str).fillna("NA") == g).to_numpy()
            yp[mask] = (y_prob[mask] >= th_map[g]).astype(int)
        return yp

    for _ in range(3):  # a few passes
        for g in groups:
            best_g = None
            for t in thresholds:
                trial = dict(current)
                trial[g] = float(t)
                y_pred = predict_with(trial)
                dp = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive))
                acc = float(accuracy_score(y_true, y_pred))
                score = (abs(dp), -acc)  # minimize |dp|, maximize acc
                if best_g is None or score < best_g[0]:
                    best_g = (score, float(t), dp, acc)
            current[g] = best_g[1]

    y_pred_final = predict_with(current)
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred_final)),
        "f1": float(f1_score(y_true, y_pred_final)),
        "demographic_parity_difference": float(demographic_parity_difference(y_true, y_pred_final, sensitive_features=sensitive)),
        "equalized_odds_difference": float(equalized_odds_difference(y_true, y_pred_final, sensitive_features=sensitive)),
    }
    return current, summary

def main(test_path: Path, model_path: Path, sensitive_col: str, out_path: Path, model_out: Path) -> None:
    paths = Paths()
    cfg = DataConfig()

    ensure_dir(paths.models_dir)
    ensure_dir(paths.reports_dir)

    df = pd.read_csv(test_path)
    blob = joblib.load(model_path)
    model = blob["model"]

    y_raw = df[cfg.target_col].astype(str)
    y_true = (y_raw.str.contains(">50K") | y_raw.str.contains("1")).astype(int).to_numpy()
    X = df.drop(columns=[cfg.target_col])

    sensitive = df[sensitive_col].astype(str).fillna("NA").to_numpy()
    y_prob = model.predict_proba(X)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 17)
    th_map, summary = group_threshold_search(y_true, y_prob, sensitive, thresholds)

    artifact = {
        "mitigation": "group_threshold_postprocessing",
        "sensitive_col": sensitive_col,
        "thresholds_by_group": th_map,
        "summary": summary,
    }

    # Save mitigation policy (threshold map) as "model artifact"
    joblib.dump({"base_model": model, "thresholds_by_group": th_map, "sensitive_col": sensitive_col}, model_out)
    save_json(artifact, out_path)

    print(f"[OK] Saved mitigation report: {out_path}")
    print(f"[OK] Saved mitigation artifact: {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=str(Paths().data_dir / "test.csv"))
    parser.add_argument("--model", type=str, default=str(Paths().models_dir / "baseline.joblib"))
    parser.add_argument("--sensitive", type=str, default=DataConfig().sensitive_cols[0])
    parser.add_argument("--out", type=str, default=str(Paths().reports_dir / "mitigation_report.json"))
    parser.add_argument("--model_out", type=str, default=str(Paths().models_dir / "mitigated_thresholds.joblib"))
    args = parser.parse_args()
    main(Path(args.test), Path(args.model), args.sensitive, Path(args.out), Path(args.model_out))
