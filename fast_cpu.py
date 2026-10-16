"""Fast CPU-only SMS spam detector.

This avoids BERT backpropagation and is intended for quick demos:
    python fast_cpu.py
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


def main():
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=15000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x_train, train_df["label"])

    y_true = test_df["label"].to_numpy()
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "specificity": float(tn / (tn + fp)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }

    with open(REPORTS / "fast_cpu_model.pkl", "wb") as file:
        pickle.dump({"model": model, "vectorizer": vectorizer}, file)
    (REPORTS / "fast_cpu_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(classification_report(
        y_true, predictions, target_names=["ham", "spam"], output_dict=True, zero_division=0
    )).transpose().to_csv(REPORTS / "fast_cpu_classification_report.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_title("Fast CPU confusion matrix")
    axes[0].set_xticks([0, 1], ["Ham", "Spam"]); axes[0].set_yticks([0, 1], ["Ham", "Spam"])
    for row in range(2):
        for column in range(2): axes[0].text(column, row, matrix[row, column], ha="center", va="center", fontsize=18)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    axes[1].plot(fpr, tpr, label=f"ROC AUC={metrics['roc_auc']:.3f}")
    axes[1].plot(recall, precision, label=f"PR AP={metrics['average_precision']:.3f}")
    axes[1].set_title("ROC and precision-recall curves"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("SMS SpamShield — Fast CPU Model", fontweight="bold")
    fig.tight_layout(); fig.savefig(REPORTS / "fast_cpu_dashboard.png", dpi=180); plt.close(fig)

    print("Fast CPU model complete")
    print(f"Features: {x_train.shape[1]:,} | Test samples: {len(y_true):,}")
    print(" | ".join(f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float)))
    print("Reports saved in reports/")


if __name__ == "__main__":
    main()
