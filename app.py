"""Hugging Face Spaces inference app.

Train first with ``python main.py --epochs 5`` so that
``reports/bert_bilstm_weights.pt`` exists.
"""

from pathlib import Path

import gradio as gr
import torch
from transformers import BertModel, BertTokenizer

from model import BertBiLSTMClassifier


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
MODEL = BertBiLSTMClassifier(BertModel.from_pretrained("bert-base-uncased"), 128, 2, 0.1)
MODEL.load_state_dict(torch.load(ROOT / "reports" / "bert_bilstm_weights.pt", map_location=DEVICE))
MODEL.to(DEVICE).eval()


def classify_sms(message):
    if not message or not message.strip():
        return "Please enter an SMS message.", 0.0
    encoded = TOKENIZER(message, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        probability = torch.sigmoid(
            MODEL(encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE))
        ).item()
    label = "SPAM" if probability >= 0.5 else "HAM"
    return f"{label} — spam confidence: {probability * 100:.2f}%", probability


demo = gr.Interface(
    fn=classify_sms,
    inputs=gr.Textbox(lines=5, label="SMS message", placeholder="Paste a suspicious message here..."),
    outputs=[gr.Textbox(label="Decision"), gr.Number(label="Spam probability")],
    title="SMS SpamShield",
    description="BERT + BiLSTM SMS spam detection demo.",
    examples=[
        ["Congratulations! You have won a free prize. Call now!"],
        ["Hey, are we still meeting for lunch today?"],
    ],
)


if __name__ == "__main__":
    demo.launch()
