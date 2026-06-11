import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
def _get_item_representation_matrix(model, item_features=None):
    """
    Returns (biases, embeddings_matrix) for items.
    embeddings_matrix shape: (n_items, no_components)
    """
    item_biases, item_emb = model.get_item_representations(item_features)
    return item_biases, item_emb

def _user_history_internal_items(interactions: csr_matrix, user_internal_id: int) -> np.ndarray:
    """
    Returns internal item indices that the user has interacted with (positive implicit interactions).
    """
    row = interactions[user_internal_id]
    return row.indices  # items with non-zero interactions

def build_user_profile_from_history(
    model,
    interactions: csr_matrix,
    user_internal_id: int,
    item_features=None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Post-hoc user profile vector = average of embeddings of user's historical interacted items.
    This enables counterfactual reasoning by removing an item from history.
    """
    _, item_emb = _get_item_representation_matrix(model, item_features)
    hist_items = _user_history_internal_items(interactions, user_internal_id)

    if hist_items.size == 0:
        # Cold start: return zero vector
        return np.zeros(item_emb.shape[1], dtype=np.float32)

    profile = item_emb[hist_items].mean(axis=0)
    if normalize:
        norm = np.linalg.norm(profile) + 1e-12
        profile = profile / norm

    return profile.astype(np.float32)

def score_items_with_profile(
    profile_vec: np.ndarray,
    item_emb: np.ndarray,
    item_bias: np.ndarray,
) -> np.ndarray:
    """
    Scores items using profile dot item_embedding + item_bias.
    """
    return profile_vec @ item_emb.T + item_bias

def topk_from_scores(scores: np.ndarray, k: int, exclude_mask: np.ndarray = None) -> np.ndarray:
    """
    Returns top-k indices by score, optionally excluding items using a boolean mask.
    """
    s = scores.copy()
    if exclude_mask is not None:
        s[exclude_mask] = -1e9
    return np.argsort(-s)[:k]

def most_influential_history_item(
    model,
    interactions: csr_matrix,
    user_internal_id: int,
    target_item_internal_id: int,
    item_features=None,
) -> Tuple[int, float]:
    """
    Finds which single history item is most responsible for the target recommendation,
    measured by cosine similarity in embedding space.
    Returns: (history_item_internal_id, similarity_score)
    """
    _, item_emb = _get_item_representation_matrix(model, item_features)
    hist_items = _user_history_internal_items(interactions, user_internal_id)

    if hist_items.size == 0:
        return -1, 0.0
        
    target_vec = item_emb[target_item_internal_id]
    target_norm = np.linalg.norm(target_vec) + 1e-12
    target_vec = target_vec / target_norm

    hist_vecs = item_emb[hist_items]
    hist_norms = np.linalg.norm(hist_vecs, axis=1) + 1e-12
    hist_vecs = hist_vecs / hist_norms[:, None]

    sims = hist_vecs @ target_vec
    best_idx = int(np.argmax(sims))
    return int(hist_items[best_idx]), float(sims[best_idx])

def counterfactual_explanation(
    model,
    interactions: csr_matrix,
    user_internal_id: int,
    k: int,
    item_features=None,
) -> Dict:
    """
    Generates a counterfactual explanation for the top-1 recommended item:

    - Original top-k recs using profile-from-history scoring
    - Identify most influential history item for top-1 rec
    - Remove that history item and recompute top-k
    - Report "If you hadn't liked X, we would recommend Y"

    Returns a dict ready to serialize to JSON.
    """
    item_bias, item_emb = _get_item_representation_matrix(model, item_features)

    # Build profile and original recommendations
    profile = build_user_profile_from_history(model, interactions, user_internal_id, item_features=item_features)
    scores = score_items_with_profile(profile, item_emb, item_bias)

    # Exclude already-known items
    known = interactions[user_internal_id].toarray().ravel() > 0
    orig_topk = topk_from_scores(scores, k=k, exclude_mask=known)
    top1 = int(orig_topk[0])

    # Find a single "most influential" history item for that top1
    infl_item, infl_sim = most_influential_history_item(
        model, interactions, user_internal_id, target_item_internal_id=top1, item_features=item_features
    )

    # If no history, no meaningful counterfactual
    if infl_item == -1:
        return {
            "user_internal_id": int(user_internal_id),
            "original_topk_internal": list(map(int, orig_topk)),
            "counterfactual_removed_history_item_internal": None,
            "counterfactual_topk_internal": list(map(int, orig_topk)),
            "reason": "No interaction history available for counterfactual.",
        }

    # Counterfactual: remove influential item from the history-based profile
    hist_items = _user_history_internal_items(interactions, user_internal_id)
    remaining = hist_items[hist_items != infl_item]

    if remaining.size == 0:
        # If removing leaves no history, profile becomes zero
        cf_profile = np.zeros(item_emb.shape[1], dtype=np.float32)
    else:
        cf_profile = item_emb[remaining].mean(axis=0)
        cf_profile = cf_profile / (np.linalg.norm(cf_profile) + 1e-12)

    cf_scores = score_items_with_profile(cf_profile, item_emb, item_bias)
    cf_topk = topk_from_scores(cf_scores, k=k, exclude_mask=known)

    return {
        "user_internal_id": int(user_internal_id),
        "original_topk_internal": list(map(int, orig_topk)),
        "top1_internal": int(top1),
        "counterfactual_removed_history_item_internal": int(infl_item),
        "influence_similarity": float(infl_sim),
        "counterfactual_topk_internal": list(map(int, cf_topk)),
        "counterfactual_top1_internal": int(cf_topk[0]),
    }
