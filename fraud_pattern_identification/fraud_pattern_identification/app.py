"""Optional dashboard:  streamlit run app.py"""
import pandas as pd
import streamlit as st
from predict import score

st.set_page_config(page_title="Fraud Pattern Identification", layout="wide")
st.title("🔍 Intelligent Fraud Pattern Identification")

up = st.sidebar.file_uploader("Upload transactions CSV", type="csv")
thr = st.sidebar.slider("Fraud threshold", 0.1, 0.95, 0.5, 0.05)
df = pd.read_csv(up) if up else pd.read_csv("data/transactions.csv")
res = score(df, thr)

c1, c2, c3 = st.columns(3)
c1.metric("Transactions", f"{len(res):,}")
c2.metric("Flagged as fraud", int(res.flagged.sum()))
c3.metric("Flag rate", f"{res.flagged.mean():.2%}")

st.subheader("Highest-risk transactions")
st.dataframe(res[["transaction_id", "customer_id", "timestamp", "amount", "country", "category",
                  "fraud_probability", "risk_level", "reasons"]].head(100), use_container_width=True)
a, b = st.columns(2)
a.subheader("Flagged by hour"); a.bar_chart(res[res.flagged == 1]["hour"].value_counts().sort_index())
b.subheader("Flagged by country"); b.bar_chart(res[res.flagged == 1]["country"].value_counts())
