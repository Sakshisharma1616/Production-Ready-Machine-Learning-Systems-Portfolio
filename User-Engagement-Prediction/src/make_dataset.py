import pandas as pd
from pathlib import Path
from src.config import RAW_DIR, PROCESSED_DIR

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = RAW_DIR / "ml-latest-small"
    ratings = pd.read_csv(base / "ml-latest-small" / "ratings.csv")
    movies = pd.read_csv(base / "ml-latest-small" / "movies.csv")
    return ratings, movies

def build_interactions(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    df = ratings.merge(movies, on="movieId", how="left")
    # timestamp is seconds since epoch
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["engaged"] = (df["rating"] >= 4.0).astype(int)

    # Light parsing of genres into list
    df["genres"] = df["genres"].fillna("(no genres listed)")
    df["genres_list"] = df["genres"].str.split("|")

    # Sort for time-aware ops
    df = df.sort_values(["userId", "ts"]).reset_index(drop=True)
    return df

def time_split_last_interaction_per_user(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # For each user, take their last interaction as test; rest as train
    idx_last = df.groupby("userId")["ts"].idxmax()
    test = df.loc[idx_last].copy()
    train = df.drop(idx_last).copy()
    return train, test

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ratings, movies = load_raw()
    df = build_interactions(ratings, movies)
    train, test = time_split_last_interaction_per_user(df)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Saved train: {train_path} | rows={len(train)}")
    print(f"Saved test : {test_path}  | rows={len(test)}")

if __name__ == "__main__":
    main()
