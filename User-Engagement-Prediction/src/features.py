import pandas as pd
import numpy as np

GENRES = [
    "Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama",
    "Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller",
    "War","Western","IMAX","(no genres listed)"
]

def add_genre_multihot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Multi-hot for genres
    for g in GENRES:
        out[f"genre__{g}"] = out["genres_list"].apply(lambda xs: int(g in xs))
    return out

def fit_aggregates(train: pd.DataFrame) -> dict:
    # user aggregates from TRAIN ONLY
    user_grp = train.groupby("userId").agg(
        user_rating_count=("rating","count"),
        user_rating_mean=("rating","mean"),
        user_engaged_rate=("engaged","mean"),
        user_last_ts=("ts","max"),
    ).reset_index()

    movie_grp = train.groupby("movieId").agg(
        movie_rating_count=("rating","count"),
        movie_rating_mean=("rating","mean"),
        movie_engaged_rate=("engaged","mean"),
    ).reset_index()

    return {"user": user_grp, "movie": movie_grp}

def apply_aggregates(df: pd.DataFrame, aggs: dict) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(aggs["user"], on="userId", how="left")
    out = out.merge(aggs["movie"], on="movieId", how="left")

    # recency feature: days since user's last TRAIN interaction
    # if user missing (shouldn't happen in this split), fill safely
    out["days_since_user_last_train"] = (
        (out["ts"] - out["user_last_ts"]).dt.total_seconds() / (3600 * 24)
    )

    # fill missing values
    for col in [
        "user_rating_count","user_rating_mean","user_engaged_rate",
        "movie_rating_count","movie_rating_mean","movie_engaged_rate",
        "days_since_user_last_train"
    ]:
        out[col] = out[col].fillna(out[col].median())

    # basic time features
    out["rating_hour"] = out["ts"].dt.hour.astype(int)
    out["rating_dayofweek"] = out["ts"].dt.dayofweek.astype(int)
    return out

def make_model_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_genre_multihot(df)
    # Keep only model columns
    keep = [
        "engaged",
        "user_rating_count","user_rating_mean","user_engaged_rate",
        "movie_rating_count","movie_rating_mean","movie_engaged_rate",
        "days_since_user_last_train",
        "rating_hour","rating_dayofweek",
    ] + [f"genre__{g}" for g in GENRES]
    return out[keep].copy()
