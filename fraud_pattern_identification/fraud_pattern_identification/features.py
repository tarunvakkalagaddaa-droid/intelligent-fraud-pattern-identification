"""Feature engineering shared by training, prediction and the dashboard."""
import numpy as np
import pandas as pd

HOME_COUNTRY = "IN"
RISKY = {"NG", "RU", "KP", "VN"}

FEATURES = [
    "amount", "log_amount", "hour", "is_night", "is_weekend", "is_online", "new_device",
    "is_foreign", "is_risky_country", "amount_vs_cust_avg", "tx_count_1h", "mins_since_last_tx",
    "category_code",
]
CATEGORY_MAP = {c: i for i, c in enumerate(
    ["grocery", "fuel", "restaurant", "electronics", "travel", "clothing", "online_shopping", "utilities"])}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].between(0, 4).astype(int)
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    df["is_foreign"] = (df["country"] != HOME_COUNTRY).astype(int)
    df["is_risky_country"] = df["country"].isin(RISKY).astype(int)
    df["category_code"] = df["category"].map(CATEGORY_MAP).fillna(-1).astype(int)

    # customer behaviour baseline (median is robust to the fraud itself)
    med = df.groupby("customer_id")["amount"].transform("median")
    df["amount_vs_cust_avg"] = df["amount"] / med.replace(0, 1)

    # velocity features
    gap = df.groupby("customer_id")["timestamp"].diff().dt.total_seconds().div(60)
    df["mins_since_last_tx"] = gap.fillna(60 * 24 * 7).clip(upper=60 * 24 * 7)
    counts = []
    for _, g in df.groupby("customer_id"):
        s = pd.Series(1, index=g["timestamp"]).rolling("1h").sum()
        counts.append(pd.Series(s.values, index=g.index))
    df["tx_count_1h"] = pd.concat(counts).sort_index()
    return df
