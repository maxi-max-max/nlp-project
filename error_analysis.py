"""
Error Analysis Script
---------------------
Analyses misclassified examples from both models and categorises
the errors into linguistic patterns (sarcasm, negation, emoji
conflict, mixed sentiment, lack of context).
Outputs: data/reports/error_analysis.json  +  data/plots/error_category_distribution.png
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# ── Simple keyword-based error categorisation ──────────────────────────
NEGATION_WORDS = {"not", "no", "never", "neither", "nobody", "nothing",
                  "nowhere", "nor", "cannot", "can't", "dont", "dont",
                  "can't", "don't", "doesn't", "doesnt",
                  "didn't", "didnt", "won't", "wont", "wouldn't", "wouldnt",
                  "shouldn't", "shouldnt", "isn't", "isnt",
                  "aren't", "arent", "wasn't", "wasnt",
                  "weren't", "werent", "haven't", "havent",
                  "hasn't", "hasnt", "without", "barely", "hardly", "scarcely"}

SARCASM_MARKERS = {"yeah right", "sure thing", "oh great", "how wonderful",
                   "so glad", "what a surprise", "totally", "/s"}

try:
    import emoji as emoji_lib
    def _has_emoji(text): return len(emoji_lib.emoji_list(text)) > 0
except ImportError:
    _EMOJI_RE = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                           "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]+")
    def _has_emoji(text): return bool(_EMOJI_RE.search(text))

SENTIMENT_WORDS_POS = {"love", "great", "good", "amazing", "awesome", "happy",
                       "best", "wonderful", "excellent", "fantastic"}
SENTIMENT_WORDS_NEG = {"hate", "bad", "terrible", "worst", "awful", "horrible",
                       "sad", "angry", "disgusting", "pathetic"}


def categorise_error(text: str, true_label: str, pred_label: str) -> str:
    lower = text.lower()
    # split() keeps contractions whole (e.g. "don't", "won't") so they match
    # the negation set directly; also check word-boundary regex for plain words
    tokens = set(lower.split())

    # Check negation
    if tokens & NEGATION_WORDS:
        return "negation"

    # Check sarcasm markers
    for marker in SARCASM_MARKERS:
        if marker in lower:
            return "sarcasm_irony"

    # Check emoji-text conflict
    if _has_emoji(text):
        return "emoji_text_conflict"

    # Check mixed sentiment (both pos and neg words present)
    has_pos = bool(tokens & SENTIMENT_WORDS_POS)
    has_neg = bool(tokens & SENTIMENT_WORDS_NEG)
    if has_pos and has_neg:
        return "mixed_sentiment"

    # Short / lacks context
    if len(tokens) <= 5:
        return "lack_of_context"

    return "other"


# ── Helpers to get predictions ─────────────────────────────────────────

TOKEN_PATTERN = re.compile(r"\b\w+\b")
ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{2,}\b")

def _count_emojis(text):
    try:
        import emoji as _e
        return len(_e.emoji_list(text))
    except ImportError:
        return 0

def _handcrafted(texts):
    rows = []
    for t in texts:
        rows.append([_count_emojis(t), t.count("!"), t.count("?"),
                      len(ALL_CAPS_PATTERN.findall(t)), len(TOKEN_PATTERN.findall(t))])
    return np.asarray(rows, dtype=np.float32)


class _TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts, self.labels, self.tok, self.ml = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.ml, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def get_logreg_preds(texts, model_path, vec_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    X_tfidf = vectorizer.transform(texts)
    X_hand = _handcrafted(texts)
    X = hstack([X_tfidf, csr_matrix(X_hand)], format="csr")
    if X.shape[1] != model.n_features_in_:
        X = X_tfidf  # fallback
    return model.predict(X)


def get_roberta_preds(texts, labels, model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    ds = _TextDS(texts, labels, tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds.append(torch.argmax(out.logits, 1).cpu().numpy())
    return np.concatenate(preds)


def analyse_model_errors(texts, y_true, y_pred, model_name):
    errors = []
    for i in range(len(texts)):
        if y_true[i] != y_pred[i]:
            true_lbl = ID_TO_LABEL[y_true[i]]
            pred_lbl = ID_TO_LABEL[y_pred[i]]
            cat = categorise_error(texts[i], true_lbl, pred_lbl)
            errors.append({
                "text": texts[i][:200],
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "error_category": cat,
            })
    # Category counts
    cats = {}
    for e in errors:
        cats[e["error_category"]] = cats.get(e["error_category"], 0) + 1
    return {
        "model": model_name,
        "total_errors": len(errors),
        "category_counts": dict(sorted(cats.items(), key=lambda x: -x[1])),
        "sample_errors": errors[:20],  # keep first 20 as examples
    }


def plot_error_categories(all_results, out_path):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    for ax, res in zip(axes, all_results):
        cats = res["category_counts"]
        ax.barh(list(cats.keys()), list(cats.values()), color="steelblue")
        ax.set_xlabel("Count")
        ax.set_title(f"Error Categories — {res['model']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved error category plot -> %s", out_path)


def main():
    log.info("Starting error analysis")

    # Load general test set
    test_df = pd.read_csv("data/train/tweets_general_test.csv")
    texts = test_df["text"].fillna("").astype(str).tolist()
    y_true = test_df["sentiment"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)

    all_results = []

    # Logistic Regression
    logreg_path = Path("models/logreg_tfidf.pkl")
    vec_path = Path("models/tfidf_vectorizer.pkl")
    if logreg_path.exists() and vec_path.exists():
        log.info("Analysing Logistic Regression errors")
        y_pred_lr = get_logreg_preds(texts, logreg_path, vec_path)
        all_results.append(analyse_model_errors(texts, y_true, y_pred_lr, "Logistic Regression"))

    # Twitter-RoBERTa
    roberta_dir = Path("models/twitter_roberta_sentiment")
    if roberta_dir.exists():
        log.info("Analysing Twitter-RoBERTa errors")
        y_pred_rob = get_roberta_preds(texts, y_true, roberta_dir)
        all_results.append(analyse_model_errors(texts, y_true, y_pred_rob, "Twitter-RoBERTa"))

    if not all_results:
        log.warning("No models found to analyse")
        return

    # Save results
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    with open("data/reports/error_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved error analysis report")

    plot_error_categories(all_results, "data/plots/error_category_distribution.png")
    log.info("Error analysis complete")


if __name__ == "__main__":
    main()
