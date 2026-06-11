import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.config import PROCESSED_DIR, MODELS_DIR, RANDOM_SEED
from src.features import fit_aggregates, apply_aggregates, make_model_table

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(PROCESSED_DIR / "train.csv", parse_dates=["ts","user_last_ts"], infer_datetime_format=True)
    # parse_dates won't parse user_last_ts because it's not in train.csv yet; fix below
    # We'll reload as raw and re-run feature pipeline properly:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    # restore datetimes
    train["ts"] = pd.to_datetime(train["ts"], utc=True)
    test["ts"] = pd.to_datetime(test["ts"], utc=True)
    aggs = fit_aggregates(train)
    train_f = apply_aggregates(train, aggs)
    test_f = apply_aggregates(test, aggs)
    X_train_df = make_model_table(train_f)
    X_test_df = make_model_table(test_f)
    y_train = X_train_df.pop("engaged")
    y_test = X_test_df.pop("engaged")

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
    ])
    model.fit(X_train_df, y_train)
    proba = model.predict_proba(X_test_df)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Test ROC-AUC: {auc:.4f}")
    artifact = {
        "model": model,
        "aggs": aggs,
        "feature_columns": list(X_train_df.columns),
    }
    out_path = MODELS_DIR / "engagement_model.joblib"
    joblib.dump(artifact, out_path)
    print(f"Saved model artifact: {out_path}")

if __name__ == "__main__":
    main()
