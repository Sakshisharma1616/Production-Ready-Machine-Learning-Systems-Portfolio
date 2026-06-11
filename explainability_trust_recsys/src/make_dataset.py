import re
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from scipy.sparse import csr_matrix
from lightfm.data import Dataset

def _read_dat(path: Path, cols: List[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="::", engine="python", names=cols, encoding="latin-1")

def load_raw(ml1m_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = _read_dat(ml1m_dir / "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
    movies  = _read_dat(ml1m_dir / "movies.dat",  ["movie_id", "title", "genres"])
    users   = _read_dat(ml1m_dir / "users.dat",   ["user_id", "gender", "age", "occupation", "zip"])
    return ratings, movies, users

def build_lightfm_dataset(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    positive_threshold: int = 4,
):
    # implicit: keep positives
    positives = ratings[ratings["rating"] >= positive_threshold].copy()
    positives["weight"] = 1.0

    # feature vocab
    movies["genres_list"] = movies["genres"].str.split("|")
    all_genres = sorted({g for row in movies["genres_list"] for g in row})

    users = users.copy()
    users["gender_feat"] = "gender=" + users["gender"].astype(str)
    users["age_feat"] = "age=" + users["age"].astype(str)
    users["occ_feat"] = "occ=" + users["occupation"].astype(str)
    user_feature_vocab = sorted(set(users["gender_feat"]) | set(users["age_feat"]) | set(users["occ_feat"]))

    # LightFM Dataset builder
    dataset = Dataset()
    dataset.fit(
        users=users["user_id"].unique(),
        items=movies["movie_id"].unique(),
        user_features=user_feature_vocab,
        item_features=all_genres,
    )

    # Build interactions
    interactions, weights = dataset.build_interactions(
        ((u, i, w) for u, i, w in positives[["user_id", "movie_id", "weight"]].itertuples(index=False))
    )

    # Build item features
    item_features = dataset.build_item_features(
        ((mid, genres) for mid, genres in movies[["movie_id", "genres_list"]].itertuples(index=False))
    )

    # Build user features
    user_features = dataset.build_user_features(
        ((uid, [gf, af, of]) for uid, gf, af, of in users[["user_id","gender_feat","age_feat","occ_feat"]].itertuples(index=False))
    )

    # Helpful mappers
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    inv_user_id_map = {v: k for k, v in user_id_map.items()}
    inv_item_id_map = {v: k for k, v in item_id_map.items()}
    inv_item_feature_map = {v: k for k, v in item_feature_map.items()}

    meta = {
        "inv_user_id_map": inv_user_id_map,
        "inv_item_id_map": inv_item_id_map,
        "inv_item_feature_map": inv_item_feature_map,
        "movies_df": movies.set_index("movie_id"),
    }

    return dataset, interactions, weights, user_features, item_features, meta
