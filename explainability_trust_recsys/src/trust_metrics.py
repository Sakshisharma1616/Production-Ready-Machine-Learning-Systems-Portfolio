import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

def item_popularity(interactions) -> np.ndarray:
    # popularity = number of users who interacted with item
    return np.asarray(interactions.sum(axis=0)).ravel()

def novelty_score(recommended_item_ids: List[int], popularity: np.ndarray) -> float:
    # Higher novelty when items are less popular: mean(-log2(pop_norm))
    pop = popularity[recommended_item_ids] + 1.0
    pop_norm = pop / pop.sum()
    return float(np.mean(-np.log2(pop_norm)))

def genre_diversity(recommended_movie_ids: List[int], movies_df) -> float:
    # simple diversity = unique genres / total genres occurrences
    genres = []
    for mid in recommended_movie_ids:
        if mid in movies_df.index:
            genres.extend(str(movies_df.loc[mid, "genres"]).split("|"))
    if not genres:
        return 0.0
    return float(len(set(genres)) / len(genres))

def list_coverage(all_recs: List[List[int]], n_items: int) -> float:
    # fraction of catalog ever shown
    shown = set()
    for recs in all_recs:
        shown.update(recs)
    return float(len(shown) / max(1, n_items))

def jaccard(a: List[int], b: List[int]) -> float:
    A, B = set(a), set(b)
    return float(len(A & B) / max(1, len(A | B)))

def stability_under_perturbation(recs_original: List[int], recs_perturbed: List[int]) -> float:
    # higher = more stable
    return jaccard(recs_original, recs_perturbed)

def summarize_trust_metrics(
    all_recs_internal: List[List[int]],
    popularity: np.ndarray,
    internal_to_movie_id,   # dict or list mapping internal item index -> external movie id
    movies_df,
) -> Dict:
    n_items = len(popularity)

    novelties = []
    diversities = []
    for recs in all_recs_internal:
        novelties.append(novelty_score(recs, popularity))
        movie_ids = [internal_to_movie_id[i] for i in recs]
        diversities.append(genre_diversity(movie_ids, movies_df))

    return {
        "catalog_coverage": list_coverage(all_recs_internal, n_items),
        "avg_novelty": float(np.mean(novelties)) if novelties else 0.0,
        "avg_genre_diversity": float(np.mean(diversities)) if diversities else 0.0,
    }
