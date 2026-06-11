import os
import json
from dataclasses import dataclass
from typing import Any, Dict
import lightgbm as lgb

@dataclass
class RegistryPaths:
    models_dir: str

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_model(model: lgb.Booster, meta: Dict[str, Any], paths: RegistryPaths) -> None:
    ensure_dir(paths.models_dir)
    model_path = os.path.join(paths.models_dir, "ltr_model.txt")
    meta_path = os.path.join(paths.models_dir, "model_meta.json")

    model.save_model(model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved model: {model_path}")
    print(f"✅ Saved meta:  {meta_path}")

def load_model(paths: RegistryPaths) -> tuple[lgb.Booster, Dict[str, Any]]:
    model_path = os.path.join(paths.models_dir, "ltr_model.txt")
    meta_path = os.path.join(paths.models_dir, "model_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata at {meta_path}")

    model = lgb.Booster(model_file=model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta
