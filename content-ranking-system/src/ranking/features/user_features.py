import pandas as pd

def add_user_aggregate_features(df: pd.DataFrame, interactions: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """
    Offline feature-store style user aggregates from interaction logs.
    Produces features that can be re-computed daily in production.
    """
    interactions = interactions.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

    # Rolling window cutoff relative to each row timestamp
    # For simplicity (and speed), we compute global last-window aggregates.
    max_ts = interactions["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=window_days)
    recent = interactions[interactions["timestamp"] >= cutoff]

    user_watch = recent.groupby("user_id")["watch_minutes"].sum().rename("u_watch_mins_30d")
    user_plays = recent.groupby("user_id")["label"].apply(lambda s: (s >= 2).sum()).rename("u_plays_30d")
    user_clicks = recent.groupby("user_id")["label"].apply(lambda s: (s >= 1).sum()).rename("u_clicks_30d")

    out = df.merge(user_watch, on="user_id", how="left") \
            .merge(user_plays, on="user_id", how="left") \
            .merge(user_clicks, on="user_id", how="left")

    for c in ["u_watch_mins_30d", "u_plays_30d", "u_clicks_30d"]:
        out[c] = out[c].fillna(0.0)

    # Simple engagement rate
    out["u_play_rate_30d"] = out["u_plays_30d"] / (out["u_clicks_30d"] + 1.0)
    return out
