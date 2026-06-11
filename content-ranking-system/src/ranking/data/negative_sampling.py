from __future__ import annotations
import numpy as np
import pandas as pd

def build_item_popularity(interactions: pd.DataFrame) -> pd.Series:
    # Popularity based on any exposure; weight higher labels more
    w = interactions["label"].clip(lower=0).astype(float)
    pop = interactions.groupby("item_id")[w.name].sum()
    pop = pop / pop.sum()
    return pop.sort_values(ascending=False)

def sample_negatives_for_user(
    user_history: set[str],
    all_items: np.ndarray,
    n: int,
    rng: np.random.Generator,
    strategy: str = "popularity",
    item_pop: pd.Series | None = None,
) -> list[str]:
    if n <= 0:
        return []

    candidates = all_items
    if strategy == "uniform":
        negs = []
        while len(negs) < n:
            it = rng.choice(candidates)
            if it not in user_history:
                negs.append(str(it))
        return negs

    if strategy == "popularity":
        if item_pop is None:
            raise ValueError("item_pop must be provided for popularity sampling")
        # sample with p ~ popularity, but avoid user_history
        items = item_pop.index.to_numpy()
        probs = item_pop.values
        negs = []
        while len(negs) < n:
            it = rng.choice(items, p=probs)
            if it not in user_history:
                negs.append(str(it))
        return negs

    raise ValueError(f"Unknown sampling strategy: {strategy}")

def make_ranking_dataset(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    negatives_per_positive: int,
    strategy: str,
    seed: int,
) -> pd.DataFrame:
    """
    Convert implicit interactions into an LTR dataset:
    - Positives: observed (user, session, item) with label in {1,2,3}
    - Negatives: sampled items per positive (label=0)
    Grouping key for ranking: (user_id, session_id)
    """
    rng = np.random.default_rng(seed)

    all_items = items["item_id"].to_numpy()
    item_pop = build_item_popularity(interactions) if strategy == "popularity" else None

    # Only keep events that indicate any engagement as positives
    positives = interactions[interactions["label"] > 0].copy()

    # Build user history (items seen/engaged)
    user_hist = interactions.groupby("user_id")["item_id"].apply(lambda s: set(s.values)).to_dict()

    rows = []
    for r in positives.itertuples(index=False):
        rows.append({
            "user_id": r.user_id,
            "session_id": r.session_id,
            "timestamp": r.timestamp,
            "device": r.device,
            "item_id": r.item_id,
            "label": int(r.label),
            "watch_minutes": int(r.watch_minutes),
            "is_negative": 0
        })

        negs = sample_negatives_for_user(
            user_history=user_hist.get(r.user_id, set()),
            all_items=all_items,
            n=negatives_per_positive,
            rng=rng,
            strategy=strategy,
            item_pop=item_pop,
        )
        for neg_it in negs:
            rows.append({
                "user_id": r.user_id,
                "session_id": r.session_id,
                "timestamp": r.timestamp,
                "device": r.device,
                "item_id": neg_it,
                "label": 0,
                "watch_minutes": 0,
                "is_negative": 1
            })

    ds = pd.DataFrame(rows)
    # Join user/item static info (feature base)
    ds = ds.merge(users, on="user_id", how="left")
    ds = ds.merge(items, on="item_id", how="left")
    return ds
