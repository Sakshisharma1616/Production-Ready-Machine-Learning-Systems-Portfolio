from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config import Paths, DataConfig
from utils import ensure_dir, save_json, save_fig

def plot_group_bar(metric_by_group: pd.Series, title: str, out_path: Path) -> None:
    plt.figure()
    metric_by_group.sort_index().plot(kind="bar")
    plt.title(title)
    plt.xlabel("Group")
    plt.ylabel("Value")
    save_fig(out_path)

def evaluate_one_sensitive(
    y_true: pd.Series, y_pred: pd.Series, sensitive: pd.Series, prefix: str, reports_dir: Path
) -> Dict[str, Any]:
    # Group-wise metric frame
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    out: Dict[str, Any] = {
        "selection_rate_by_group": mf.by_group["selection_rate"].to_dict(),
        "tpr_by_group": mf.by_group["tpr"].to_dict(),
        "fpr_by_group": mf.by_group["fpr"].to_dict(),
        "demographic_parity_difference": float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)),
        "demographic_parity_ratio": float(demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive)),
        "equalized_odds_difference": float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)),
    }

    # Plots
    plot_group_bar(mf.by_group["selection_rate"], f"{prefix}: Selection Rate by Group", reports_dir / f"{prefix}_selection_rate.png")
    plot_group_bar(mf.by_group["tpr"], f"{prefix}: True Positive Rate by Group", reports_dir / f"{prefix}_tpr.png")
    plot_group_bar(mf.by_group["fpr"], f"{prefix}: False Positive Rate by Group", reports_dir / f"{prefix}_fpr.png")

    return out

def main(test_path: Path, model_path: Path, out_json: Path) -> None:
    paths = Paths()
    cfg = DataConfig()
    ensure_dir(paths.reports_dir)
    df = pd.read_csv(test_path)
    blob = joblib.load(model_path)
    model = blob["model"]
    y_raw = df[cfg.target_col].astype(str)
    y_true = (y_raw.str.contains(">50K") | y_raw.str.contains("1")).astype(int)
    X = df.drop(columns=[cfg.target_col])

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold": 0.5,
    }

    fairness: Dict[str, Any] = {}
    for s in cfg.sensitive_cols:
        fairness[s] = evaluate_one_sensitive(
            y_true=y_true,
            y_pred=pd.Series(y_pred, index=df.index),
            sensitive=df[s].astype(str).fillna("NA"),
            prefix=f"sensitive_{s}",
            reports_dir=paths.reports_dir,
        )

    # Overall confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Overall)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for (i, j), val in zip([(0,0),(0,1),(1,0),(1,1)], cm.flatten()):
        plt.text(j, i, str(val), ha="center", va="center")
    save_fig(paths.reports_dir / "confusion_matrix_overall.png")

    results = {"overall": overall, "fairness": fairness}
    save_json(results, out_json)

    print(f"[OK] Saved evaluation: {out_json}")
    print(f"[OK] Plots saved under: {paths.reports_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=str(Paths().data_dir / "test.csv"))
    parser.add_argument("--model", type=str, default=str(Paths().models_dir / "baseline.joblib"))
    parser.add_argument("--out", type=str, default=str(Paths().reports_dir / "eval_baseline.json"))
    args = parser.parse_args()
    main(Path(args.test), Path(args.model), Path(args.out))
