"""Generate a synthetic credit-card transaction dataset with embedded fraud patterns.

Fraud patterns injected:
  1. High-value purchases far above the customer's usual spend
  2. Late-night transactions (00:00-05:00)
  3. Rapid-fire bursts (several transactions within minutes)
  4. Foreign / high-risk country transactions
  5. Online purchases from a new device
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
CATEGORIES = ["grocery", "fuel", "restaurant", "electronics", "travel", "clothing", "online_shopping", "utilities"]
COUNTRIES = ["IN", "IN", "IN", "IN", "IN", "US", "UK", "SG", "AE"]
RISKY_COUNTRIES = ["NG", "RU", "KP", "VN"]


def generate(n_customers=500, n_tx=20000, fraud_rate=0.03, path="data/transactions.csv"):
    start = pd.Timestamp("2026-01-01")
    cust_avg = RNG.gamma(4, 25, n_customers) + 20  # typical spend per customer

    rows = []
    for i in range(n_tx):
        c = int(RNG.integers(0, n_customers))
        ts = start + pd.Timedelta(seconds=int(RNG.integers(0, 90 * 24 * 3600)))
        # normal behaviour: mostly daytime
        hour = int(np.clip(RNG.normal(14, 4), 6, 23))
        ts = ts.replace(hour=hour, minute=int(RNG.integers(0, 60)))
        rows.append(dict(
            transaction_id=i, customer_id=c, timestamp=ts,
            amount=round(float(RNG.lognormal(np.log(cust_avg[c]), 0.5)), 2),
            category=RNG.choice(CATEGORIES),
            country=RNG.choice(COUNTRIES),
            is_online=int(RNG.random() < 0.35),
            new_device=int(RNG.random() < 0.03),
            is_fraud=0,
        ))
    df = pd.DataFrame(rows)

    # inject fraud
    n_fraud = int(n_tx * fraud_rate)
    idx = RNG.choice(df.index, n_fraud, replace=False)
    for j, k in enumerate(idx):
        pattern = j % 5
        df.loc[k, "is_fraud"] = 1
        c = df.loc[k, "customer_id"]
        if pattern == 0:      # high value
            df.loc[k, "amount"] = round(cust_avg[c] * RNG.uniform(8, 25), 2)
            df.loc[k, "category"] = RNG.choice(["electronics", "travel"])
        elif pattern == 1:    # late night
            ts = df.loc[k, "timestamp"].replace(hour=int(RNG.integers(0, 5)))
            df.loc[k, "timestamp"] = ts
            df.loc[k, "amount"] = round(cust_avg[c] * RNG.uniform(2, 6), 2)
        elif pattern == 2:    # burst: add 3 more quick transactions after this one
            base = df.loc[k, "timestamp"]
            for b in range(1, 4):
                new = df.loc[k].copy()
                new["timestamp"] = base + pd.Timedelta(minutes=b * int(RNG.integers(1, 4)))
                new["amount"] = round(float(RNG.uniform(50, 300)), 2)
                new["transaction_id"] = len(df) + len(rows) + j * 10 + b
                df = pd.concat([df, new.to_frame().T], ignore_index=True)
        elif pattern == 3:    # risky country
            df.loc[k, "country"] = RNG.choice(RISKY_COUNTRIES)
            df.loc[k, "amount"] = round(cust_avg[c] * RNG.uniform(3, 10), 2)
        else:                 # online + new device
            df.loc[k, "is_online"] = 1
            df.loc[k, "new_device"] = 1
            df.loc[k, "amount"] = round(cust_avg[c] * RNG.uniform(3, 8), 2)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = df.index
    for col in ["amount"]:
        df[col] = df[col].astype(float)
    for col in ["customer_id", "is_online", "new_device", "is_fraud"]:
        df[col] = df[col].astype(int)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} transactions ({df.is_fraud.sum()} fraud, {df.is_fraud.mean():.2%}) -> {path}")
    return df


if __name__ == "__main__":
    generate()
