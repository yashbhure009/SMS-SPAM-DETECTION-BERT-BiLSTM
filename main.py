"""Hackathon-ready SMS spam detection demo."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertModel

from data import SMSDataset
from model import BertBiLSTMClassifier

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
SEED = 42
EPOCHS = 5
VERBOSE = 2


def print_section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    print_section("1. DATASET INSPECTION")
    df = pd.read_csv(ROOT / "spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna().reset_index(drop=True)
    print(f"Total messages: {len(df):,}")
    print(f"Class counts: {df['label'].value_counts().to_dict()}")
    print(f"Spam rate: {df['label'].mean() * 100:.2f}%")
    print(f"Average message length: {df['text'].str.len().mean():.1f} characters")
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])


def metric_dict(y_true, predictions, probabilities):
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def evaluate_predictions(y_true, probabilities, name):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    metrics = metric_dict(y_true, predictions, probabilities)
    report = classification_report(y_true, predictions, target_names=["ham", "spam"],output_dict=True, zero_division=0)
    with open(REPORTS / f"{name}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame(report).transpose().to_csv(REPORTS / f"{name}_classification_report.csv")
    return predictions, metrics


def train_bert(train_df, test_df, epochs, batch_size, verbose=VERBOSE):
    print_section("2. BERT + BiLSTM TRAINING")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pin = torch.cuda.is_available()
    train_loader = DataLoader(SMSDataset(train_df, tokenizer, 128), batch_size=batch_size, shuffle=True, pin_memory=pin)
    test_loader = DataLoader(SMSDataset(test_df, tokenizer, 128), batch_size=batch_size, pin_memory=pin)
    model = BertBiLSTMClassifier(BertModel.from_pretrained("bert-base-uncased"), 128, 2, 0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    use_amp = device.type == "cuda"  # mixed precision: faster + less memory on T4
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Device: {device} | Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    history = {"train_loss": [], "test_loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    final_probabilities, final_labels = None, None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids, mask, labels = (batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device))
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(ids, mask).squeeze(-1)
            loss = criterion(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        test_loss, probabilities, labels_seen = 0.0, [], []
        with torch.no_grad():
            for batch in test_loader:
                ids, mask, labels = (batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device))
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(ids, mask).squeeze(-1)
                logits = logits.float()
                test_loss += criterion(logits, labels).item()
                probabilities.extend(torch.sigmoid(logits).cpu().numpy().astype(float))
                labels_seen.extend(labels.cpu().numpy().astype(int))
        _, metrics = evaluate_predictions(labels_seen, probabilities, "bert_bilstm")
        history["train_loss"].append(train_loss / len(train_loader))
        history["test_loss"].append(test_loss / len(test_loader))
        history["accuracy"].append(metrics["accuracy"]); history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"]); history["f1"].append(metrics["f1_score"])
        final_probabilities, final_labels = np.array(probabilities), np.array(labels_seen)
        if verbose == 1:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={history['train_loss'][-1]:.4f} | "
                f"test_loss={history['test_loss'][-1]:.4f} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"precision={metrics['precision']:.4f} | "
                f"recall={metrics['recall']:.4f} | "
                f"f1={metrics['f1_score']:.4f}"
            )
        elif verbose == 2:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"loss: {history['train_loss'][-1]:.4f} - "
                f"val_loss: {history['test_loss'][-1]:.4f} - "
                f"val_f1: {metrics['f1_score']:.4f}"
            )

    torch.save(model.state_dict(), REPORTS / "bert_bilstm_weights.pt")
    return model, tokenizer, history, final_labels, final_probabilities


def train_decision_tree(train_df, test_df):
    print_section("3. EXPLAINABLE DECISION-TREE BASELINE")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)
    train_features = vectorizer.fit_transform(train_df["text"])
    test_features = vectorizer.transform(test_df["text"])
    tree = DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=SEED)
    tree.fit(train_features, train_df["label"])
    probabilities = tree.predict_proba(test_features)[:, 1]
    _, metrics = evaluate_predictions(test_df["label"].to_numpy(), probabilities, "decision_tree")
    print(f"Decision tree depth: {tree.get_depth()} | features: {train_features.shape[1]:,}")
    print(" | ".join(f"{key}={value:.4f}" for key, value in metrics.items() if isinstance(value, float)))
    plt.figure(figsize=(22, 12))
    plot_tree(tree, max_depth=3, feature_names=vectorizer.get_feature_names_out(), class_names=["ham", "spam"], filled=True, rounded=True, fontsize=7)
    plt.title("Explainable Decision Tree (first 3 levels)"); plt.tight_layout(); plt.savefig(REPORTS / "decision_tree.png", dpi=180); plt.close()
    return metrics


def save_visualizations(labels, probabilities, history):
    print_section("4. VISUAL REPORT GENERATION")
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes[0, 0].plot(history["train_loss"], marker="o", label="Train loss"); axes[0, 0].plot(history["test_loss"], marker="o", label="Test loss")
    axes[0, 0].set_title("Training and validation loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    for key in ("accuracy", "precision", "recall", "f1"):
        axes[0, 1].plot(history[key], marker="o", label=key.upper())
    axes[0, 1].set_ylim(0, 1.05); axes[0, 1].set_title("Metrics by epoch"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    im = axes[1, 0].imshow(matrix, cmap="Blues"); axes[1, 0].set_title("Confusion matrix (BERT + BiLSTM)")
    axes[1, 0].set_xticks([0, 1], ["Ham", "Spam"]); axes[1, 0].set_yticks([0, 1], ["Ham", "Spam"])
    for row in range(2):
        for column in range(2): axes[1, 0].text(column, row, matrix[row, column], ha="center", va="center", fontsize=18)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    fpr, tpr, _ = roc_curve(labels, probabilities); precision, recall, _ = precision_recall_curve(labels, probabilities)
    axes[1, 1].plot(fpr, tpr, label=f"ROC AUC={roc_auc_score(labels, probabilities):.3f}")
    axes[1, 1].plot(recall, precision, label=f"PR AP={average_precision_score(labels, probabilities):.3f}")
    axes[1, 1].plot([0, 1], [0, 1], "--", alpha=0.4); axes[1, 1].set_title("ROC and precision-recall curves")
    axes[1, 1].set_xlabel("FPR / Recall"); axes[1, 1].set_ylabel("TPR / Precision"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    fig.suptitle("SMS SpamShield — Evaluation Dashboard", fontsize=18, fontweight="bold"); fig.tight_layout(); fig.savefig(REPORTS / "evaluation_dashboard.png", dpi=180); plt.close(fig)
    print("Saved evaluation_dashboard.png, decision_tree.png, metrics JSON/CSV, and model weights")


def predict_message(message, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(message, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        logit = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device)).float()
        probability = torch.sigmoid(logit).item()
    label = "SPAM" if probability >= 0.5 else "HAM"
    print(f"\nInput: {message}\nDecision: {label}\nSpam confidence: {probability * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Hackathon-ready SMS spam detector")
    parser.add_argument("--message", help="Classify one SMS after training")
    parser.add_argument("--interactive", action="store_true", help="Classify messages until 'quit'")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of BERT training epochs")
    parser.add_argument(
        "--verbose", type=int, choices=[0, 1, 2], default=VERBOSE,
        help="Training output: 0=silent, 1=all metrics, 2=one summary line per epoch",
    )
    args = parser.parse_args()
    set_seed(); REPORTS.mkdir(exist_ok=True)
    train_df, test_df = load_data()
    model, tokenizer, history, labels, probabilities = train_bert(
        train_df, test_df, args.epochs, 32, args.verbose
    )
    tree_metrics = train_decision_tree(train_df, test_df)
    save_visualizations(labels, probabilities, history)
    bert_metrics = json.loads((REPORTS / "bert_bilstm_metrics.json").read_text(encoding="utf-8"))
    comparison = pd.DataFrame({"BERT + BiLSTM": bert_metrics, "Decision Tree": tree_metrics}).transpose()
    comparison.to_csv(REPORTS / "model_comparison.csv")
    print_section("5. FINAL SCORECARD")
    print(comparison[["accuracy", "precision", "recall", "f1_score", "roc_auc", "specificity"]].round(4).to_string())
    device = next(model.parameters()).device
    if args.message:
        predict_message(args.message, model, tokenizer, device)
    if args.interactive:
        print("\nInteractive mode — type an SMS, or 'quit' to stop.")
        while True:
            message = input("SMS> ").strip()
            if message.lower() in {"quit", "exit"}: break
            if message: predict_message(message, model, tokenizer, device)


if __name__ == "__main__":
    main()