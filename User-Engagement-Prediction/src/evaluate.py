import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from src.config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR
from src.features import apply_aggregates, make_model_table

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(MODELS_DIR / "engagement_model.joblib")
    model = artifact["model"]
    aggs = artifact["aggs"]
    cols = artifact["feature_columns"]

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    test["ts"] = pd.to_datetime(test["ts"], utc=True)

    test_f = apply_aggregates(test, aggs)
    X_test = make_model_table(test_f)
    y_test = X_test.pop("engaged")
    X_test = X_test[cols]

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    metrics_path = REPORTS_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"ROC-AUC: {auc:.6f}\n")
        f.write(f"PR-AUC : {ap:.6f}\n")
    print(f"Saved metrics: {metrics_path}")

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("ROC Curve - Engagement Prediction")
    roc_path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()

    # PR curve
    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title("Precision-Recall Curve - Engagement Prediction")
    pr_path = REPORTS_DIR / "pr_curve.png"
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plots: {roc_path}, {pr_path}")

if __name__ == "__main__":
    main()
