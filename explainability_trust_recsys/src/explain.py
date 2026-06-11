import numpy as np
import pandas as pd
from typing import Dict, Tuple

def _sparse_row_indices_and_data(sparse_row):
    indices = sparse_row.indices
    data = sparse_row.data
    return indices, data

def explain_recommendation(
    model,
    user_internal_id: int,
    item_internal_id: int,
    user_features,
    item_features,
    inv_item_feature_map: Dict[int, str],
    top_n_features: int = 5,
) -> Dict:
    """
    Feature-attribution explanation:
    - predicted score
    - top contributing item features (e.g., genres)
    """

    uvec = model.get_user_representations(user_features)[1][user_internal_id]
    ivec = model.get_item_representations(item_features)[1][item_internal_id]

    score = float(np.dot(uvec, ivec))

    item_row = item_features[item_internal_id]
    f_idx, f_val = _sparse_row_indices_and_data(item_row)

    feat_emb = model.item_embeddings  # feature embeddings

    contribs = []
    for idx, val in zip(f_idx, f_val):
        f_name = inv_item_feature_map.get(int(idx), f"feature_{idx}")
        c = float(np.dot(uvec, feat_emb[int(idx)]) * float(val))
        contribs.append((f_name, c))

    contribs.sort(key=lambda x: -abs(x[1]))
    top = contribs[:top_n_features]

    return {
        "user_internal_id": int(user_internal_id),
        "item_internal_id": int(item_internal_id),
        "predicted_score": float(score),
        "top_item_feature_contributions": [{"feature": f, "contribution": c} for f, c in top],
    }

def movie_title(movie_id: int, movies_df: pd.DataFrame) -> str:
    if movies_df is None:
        return str(movie_id)
    if movie_id in movies_df.index:
        return str(movies_df.loc[movie_id, "title"])
    return str(movie_id)

def format_feature_explanation(expl: Dict, item_movie_id: int, movies_df: pd.DataFrame) -> str:
    title = movie_title(item_movie_id, movies_df)
    lines = [
        f"Recommended: {title}",
        f"Predicted affinity score: {expl['predicted_score']:.4f}",
        "Top reasons (feature contributions):",
    ]
    for r in expl["top_item_feature_contributions"]:
        lines.append(f"  - {r['feature']}: {r['contribution']:+.4f}")
    return "\n".join(lines)
