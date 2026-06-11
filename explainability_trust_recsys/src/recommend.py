import numpy as np

def recommend_for_user(
    model,
    user_internal_id: int,
    interactions,
    user_features,
    item_features,
    k: int = 10,
):
    n_users, n_items = interactions.shape
    scores = model.predict(
        user_ids=user_internal_id,
        item_ids=np.arange(n_items),
        user_features=user_features,
        item_features=item_features,
        num_threads=4,
    )

    # Remove already interacted items
    known_items = interactions[user_internal_id].toarray().ravel() > 0
    scores[known_items] = -1e9

    top_items = np.argsort(-scores)[:k]
    return top_items, scores[top_items]
