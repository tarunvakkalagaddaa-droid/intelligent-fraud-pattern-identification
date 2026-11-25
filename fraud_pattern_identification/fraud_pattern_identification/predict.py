"""Score a CSV of transactions:  python predict.py data/transactions.csv"""
import sys
import joblib
import pandas as pd
from features import FEATURES, build_features


def score(df_raw: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    m = joblib.load("models/fraud_models.joblib")
    df = build_features(df_raw)
    df["fraud_probability"] = m["rf"].predict_proba(df[FEATURES])[:, 1]
    df["anomaly_score"] = -m["iso"].score_samples(df[FEATURES])
    df["flagged"] = (df["fraud_probability"] >= threshold).astype(int)
    df["risk_level"] = pd.cut(df["fraud_probability"], [-1, .3, .7, 1.01], labels=["Low", "Medium", "High"])
    df["reasons"] = df.apply(explain, axis=1)
    return df.sort_values("fraud_probability", ascending=False)


def explain(r) -> str:
    out = []
    if r.amount_vs_cust_avg > 5: out.append("amount far above customer's norm")
    if r.is_night: out.append("late-night transaction")
    if r.tx_count_1h >= 3: out.append("rapid burst of transactions")
    if r.is_risky_country: out.append("high-risk country")
    elif r.is_foreign: out.append("foreign transaction")
    if r.is_online and r.new_device: out.append("online purchase from new device")
    return "; ".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/transactions.csv"
    res = score(pd.read_csv(path))
    res.to_csv("outputs/scored_transactions.csv", index=False)
    print(res[["transaction_id", "customer_id", "amount", "fraud_probability", "risk_level", "reasons"]].head(15).to_string(index=False))
    print(f"\nFlagged {res.flagged.sum()} of {len(res)} transactions -> outputs/scored_transactions.csv")
