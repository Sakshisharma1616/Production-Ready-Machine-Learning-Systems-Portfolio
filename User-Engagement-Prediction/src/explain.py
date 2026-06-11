import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
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

    X = make_model_table(test_f)
    y = X.pop("engaged")
    X = X[cols]

    # --- Permutation Importance (model-agnostic, Netflix-friendly)
    r = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="roc_auc")
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)

    top = imp.head(20)[::-1]
    plt.figure()
    top.plot(kind="barh")
    plt.title("Top 20 Features - Permutation Importance (ROC-AUC)")
    out_path = REPORTS_DIR / "perm_importance_top20.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # --- SHAP (for Logistic Regression: use LinearExplainer)
    # Notes: With pipelines, we need transformed data. We'll use the pipeline steps directly.
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    X_scaled = scaler.transform(X)
    explainer = shap.LinearExplainer(clf, X_scaled)
    shap_values = explainer(X_scaled)
    shap.summary_plot(shap_values, features=X, show=False, max_display=20)
    shap_path = REPORTS_DIR / "shap_summary_top20.png"
    plt.savefig(shap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {shap_path}")

if __name__ == "__main__":
    main()
