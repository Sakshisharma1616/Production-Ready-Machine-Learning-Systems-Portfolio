import pandas as pd

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Context = request-time features (device, time-of-day, day-of-week).
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    out["hour"] = out["timestamp"].dt.hour.astype(int)
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype(int)  # 0=Mon

    # Simple bucketization
    out["is_prime_time"] = out["hour"].between(19, 23).astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out
