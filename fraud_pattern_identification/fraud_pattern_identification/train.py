"""Train + evaluate fraud models, then mine human-readable fraud patterns."""
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, average_precision_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from features import FEATURES, build_features

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def main(path="data/transactions.csv"):
    raw = pd.read_csv(path)
    df = build_features(raw)
    X, y = df[FEATURES], df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # 1) Supervised model: Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("=== Random Forest (supervised) ===")
    print(classification_report(y_te, pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_te, proba), 4),
          "| PR-AUC:", round(average_precision_score(y_te, proba), 4))

    # 2) Unsupervised model: Isolation Forest (finds NEW / unlabeled anomalies)
    iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()), random_state=42)
    iso.fit(X_tr)
    iso_pred = (iso.predict(X_te) == -1).astype(int)
    iso_score = -iso.score_samples(X_te)
    print("=== Isolation Forest (unsupervised) ===")
    print(classification_report(y_te, iso_pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_te, iso_score), 4))

    # 3) Pattern identification: shallow decision tree -> readable rules
    tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", min_samples_leaf=20, random_state=42)
    tree.fit(X_tr, y_tr)
    rules = export_text(tree, feature_names=FEATURES)
    with open("outputs/fraud_rules.txt", "w") as f:
        f.write("Fraud pattern rules learned by a shallow decision tree\n" + "=" * 55 + "\n" + rules)
    print("Top fraud rules saved to outputs/fraud_rules.txt")

    # Charts
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(figsize=(7, 5), color="#2563eb", title="Feature importance (fraud patterns)")
    plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=120); plt.close()

    ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["Legit", "Fraud"]).plot(cmap="Blues")
    plt.title("Random Forest confusion matrix"); plt.savefig("outputs/confusion_matrix.png", dpi=120); plt.close()

    p, r, _ = precision_recall_curve(y_te, proba)
    plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curve")
    plt.savefig("outputs/pr_curve.png", dpi=120); plt.close()

    fraud = df[df.is_fraud == 1]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    fraud["hour"].value_counts().sort_index().plot.bar(ax=ax[0], title="Fraud by hour")
    fraud["country"].value_counts().plot.bar(ax=ax[1], title="Fraud by country")
    fraud["category"].value_counts().plot.bar(ax=ax[2], title="Fraud by category")
    plt.tight_layout(); plt.savefig("outputs/fraud_patterns.png", dpi=120); plt.close()

    joblib.dump({"rf": rf, "iso": iso}, "models/fraud_models.joblib")
    print("Models saved to models/fraud_models.joblib; charts saved to outputs/")


if __name__ == "__main__":
    main()
