import argparse
import os
import json
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

@dataclass
class Config:
    seed: int
    raw_dir: str
    n_users: int
    n_items: int
    n_interactions: int
    n_sessions: int
    start_date: str
    end_date: str

def _load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["paths"]["raw_dir"]
    return Config(
        seed=int(cfg["project"]["seed"]),
        raw_dir=raw_dir,
        n_users=int(cfg["data_gen"]["n_users"]),
        n_items=int(cfg["data_gen"]["n_items"]),
        n_interactions=int(cfg["data_gen"]["n_interactions"]),
        n_sessions=int(cfg["data_gen"]["n_sessions"]),
        start_date=str(cfg["data_gen"]["start_date"]),
        end_date=str(cfg["data_gen"]["end_date"]),
    )

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def main(config_path: str) -> None:
    cfg = _load_config(config_path)
    np.random.seed(cfg.seed)
    _ensure_dir(cfg.raw_dir)

    # Users
    user_ids = [f"u_{i:04d}" for i in range(cfg.n_users)]
    age_bucket = np.random.choice(["18-24", "25-34", "35-44", "45-54", "55+"], size=cfg.n_users, p=[0.18, 0.34, 0.22, 0.16, 0.10])
    country = np.random.choice(["US", "IN", "BR", "GB", "CA", "DE", "JP"], size=cfg.n_users, p=[0.35, 0.18, 0.12, 0.10, 0.08, 0.08, 0.09])
    is_kids = np.random.binomial(1, 0.12, size=cfg.n_users)

    users = pd.DataFrame({
        "user_id": user_ids,
        "age_bucket": age_bucket,
        "country": country,
        "is_kids_profile": is_kids.astype(int)
    })

    # Items
    item_ids = [f"i_{i:04d}" for i in range(cfg.n_items)]
    genres = ["Drama", "Comedy", "Action", "Thriller", "Romance", "SciFi", "Horror", "Doc", "Kids"]
    genre = np.random.choice(genres, size=cfg.n_items, p=[0.20, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.06, 0.06])
    maturity = np.random.choice(["G", "PG", "PG-13", "R"], size=cfg.n_items, p=[0.10, 0.25, 0.40, 0.25])
    release_year = np.random.randint(1990, 2026, size=cfg.n_items)
    runtime_min = np.random.randint(20, 160, size=cfg.n_items)

    items = pd.DataFrame({
        "item_id": item_ids,
        "genre": genre,
        "maturity": maturity,
        "release_year": release_year,
        "runtime_min": runtime_min
    })

    # Sessions + timestamps
    start = datetime.fromisoformat(cfg.start_date)
    end = datetime.fromisoformat(cfg.end_date)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    timestamps = np.random.randint(start_ts, end_ts + 1, size=cfg.n_interactions)
    timestamps = pd.to_datetime(timestamps, unit="s")

    session_ids = [f"s_{i:05d}" for i in range(cfg.n_sessions)]
    interaction_sessions = np.random.choice(session_ids, size=cfg.n_interactions, replace=True)

    # Popularity skew (Zipf) to mimic blockbuster-heavy catalog traffic
    item_pop = np.random.zipf(a=1.4, size=cfg.n_items).astype(float)
    item_pop = item_pop / item_pop.sum()
    chosen_items = np.random.choice(item_ids, size=cfg.n_interactions, p=item_pop)
    chosen_users = np.random.choice(user_ids, size=cfg.n_interactions, replace=True)

    # Implicit signals: impression -> click/play -> watch_time bucket
    # We'll generate a "relevance label" 0/1/2/3 typical LTR grading.
    base_click = 0.08
    base_play = 0.05

    # Genre affinity by age bucket (simple synthetic signal)
    genre_affinity = {
        "18-24": {"Action": 1.15, "Comedy": 1.10, "SciFi": 1.10, "Drama": 1.00, "Kids": 0.60},
        "25-34": {"Drama": 1.10, "Thriller": 1.10, "Action": 1.05, "Doc": 1.00, "Kids": 0.70},
        "35-44": {"Drama": 1.10, "Doc": 1.10, "Romance": 1.05, "Action": 0.95, "Horror": 0.85},
        "45-54": {"Drama": 1.10, "Doc": 1.15, "Thriller": 1.00, "Comedy": 0.95, "Action": 0.90},
        "55+":   {"Doc": 1.20, "Drama": 1.05, "Comedy": 0.95, "Action": 0.85, "Horror": 0.70},
    }

    users_idx = users.set_index("user_id")
    items_idx = items.set_index("item_id")

    labels = []
    watch_mins = []
    devices = np.random.choice(["mobile", "tv", "web", "tablet"], size=cfg.n_interactions, p=[0.30, 0.45, 0.20, 0.05])

    for u, it in zip(chosen_users, chosen_items):
        age = users_idx.loc[u, "age_bucket"]
        g = items_idx.loc[it, "genre"]
        kids = int(users_idx.loc[u, "is_kids_profile"])
        mat = items_idx.loc[it, "maturity"]

        aff = genre_affinity.get(age, {}).get(g, 1.0)
        if kids == 1 and g != "Kids":
            aff *= 0.75
        if kids == 1 and mat in ["PG-13", "R"]:
            aff *= 0.55

        p_click = min(0.5, base_click * aff * 1.2)
        clicked = np.random.rand() < p_click

        p_play = min(0.4, base_play * aff * (1.5 if clicked else 0.9))
        played = np.random.rand() < p_play

        if not clicked and not played:
            label = 0
            w = 0
        elif clicked and not played:
            label = 1
            w = int(np.random.exponential(scale=2))
        elif played:
            # watch time correlated with affinity
            w = int(np.random.exponential(scale=15 * aff))
            label = 2 if w < 10 else 3

        labels.append(label)
        watch_mins.append(w)

    interactions = pd.DataFrame({
        "user_id": chosen_users,
        "item_id": chosen_items,
        "session_id": interaction_sessions,
        "timestamp": timestamps,
        "device": devices,
        "label": labels,
        "watch_minutes": watch_mins
    }).sort_values("timestamp").reset_index(drop=True)

    users.to_csv(os.path.join(cfg.raw_dir, "users.csv"), index=False)
    items.to_csv(os.path.join(cfg.raw_dir, "items.csv"), index=False)
    interactions.to_csv(os.path.join(cfg.raw_dir, "interactions.csv"), index=False)

    summary = {
        "n_users": int(cfg.n_users),
        "n_items": int(cfg.n_items),
        "n_interactions": int(cfg.n_interactions),
        "label_counts": interactions["label"].value_counts().to_dict(),
        "time_range": [str(interactions["timestamp"].min()), str(interactions["timestamp"].max())]
    }
    with open(os.path.join(cfg.raw_dir, "data_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ Generated raw data in:", cfg.raw_dir)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/ranker.yaml")
    args = ap.parse_args()
    main(args.config)
