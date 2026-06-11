import joblib
import numpy as np
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, auc_score
from src.config import MODELS_DIR, RANDOM_SEED, NO_COMPONENTS, LEARNING_RATE, EPOCHS, NUM_THREADS

def train_lightfm(interactions, user_features, item_features):
    np.random.seed(RANDOM_SEED)

    train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=RANDOM_SEED)

    model = LightFM(
        no_components=NO_COMPONENTS,
        learning_rate=LEARNING_RATE,
        loss="warp",
        random_state=RANDOM_SEED,
    )

    model.fit(
        train,
        user_features=user_features,
        item_features=item_features,
        epochs=EPOCHS,
        num_threads=NUM_THREADS,
        verbose=True,
    )

    prec = precision_at_k(model, test, user_features=user_features, item_features=item_features, k=10).mean()
    auc  = auc_score(model, test, user_features=user_features, item_features=item_features).mean()

    return model, {"precision@10": float(prec), "auc": float(auc)}

def save_model(model, metrics: dict, path_prefix: str = "lightfm"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{path_prefix}.joblib")
    joblib.dump(metrics, MODELS_DIR / f"{path_prefix}_metrics.joblib")
