import pandas as pd

def add_item_aggregate_features(df: pd.DataFrame, interactions: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    interactions = interactions.copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
    max_ts = interactions["timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=window_days)
    recent = interactions[interactions["timestamp"] >= cutoff]

    item_watch = recent.groupby("item_id")["watch_minutes"].sum().rename("i_watch_mins_30d")
    item_plays = recent.groupby("item_id")["label"].apply(lambda s: (s >= 2).sum()).rename("i_plays_30d")
    item_clicks = recent.groupby("item_id")["label"].apply(lambda s: (s >= 1).sum()).rename("i_clicks_30d")

    out = df.merge(item_watch, on="item_id", how="left") \
            .merge(item_plays, on="item_id", how="left") \
            .merge(item_clicks, on="item_id", how="left")

    for c in ["i_watch_mins_30d", "i_plays_30d", "i_clicks_30d"]:
        out[c] = out[c].fillna(0.0)

    out["i_play_rate_30d"] = out["i_plays_30d"] / (out["i_clicks_30d"] + 1.0)
    return out
