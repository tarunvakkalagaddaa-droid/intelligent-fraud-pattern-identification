# Intelligent Fraud Pattern Identification

A simple, end-to-end machine-learning project that detects fraudulent card transactions and
surfaces the **patterns** behind them.

## What it does
| Step | File | Description |
|------|------|-------------|
| 1 | `generate_data.py` | Creates synthetic transactions with 5 fraud patterns (high value, late night, rapid bursts, risky country, new device online) |
| 2 | `features.py` | Feature engineering: amount vs customer norm, velocity (tx in last hour), night/foreign flags, etc. |
| 3 | `train.py` | Trains **Random Forest** (supervised) + **Isolation Forest** (unsupervised anomaly detection), evaluates, mines readable rules with a **Decision Tree**, saves charts |
| 4 | `predict.py` | Scores any CSV, gives fraud probability, risk level and human-readable reasons |
| 5 | `app.py` | Optional Streamlit dashboard |

## Setup & Run
```bash
pip install -r requirements.txt
python generate_data.py     # -> data/transactions.csv
python train.py             # -> models/, outputs/ (charts, fraud_rules.txt)
python predict.py           # -> outputs/scored_transactions.csv
streamlit run app.py        # optional dashboard
```

## Using your own data
Provide a CSV with columns: `transaction_id, customer_id, timestamp, amount, category, country, is_online, new_device`
(and `is_fraud` for training). Edit `features.py` to add more signals.

## Techniques
- Class-imbalance handling (`class_weight`), stratified split
- Supervised + unsupervised detection (catches known and unknown fraud)
- Interpretable pattern mining via decision-tree rules and feature importance
