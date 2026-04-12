import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack, load_npz
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluate_models.log"),
    ],
)
log = logging.getLogger(__name__)


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_IDS = [LABEL_TO_ID[x] for x in LABEL_ORDER]

TOKEN_PATTERN = re.compile(r"\b\w+\b")
ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
EMOJI_FALLBACK_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

try:
    import emoji

    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    log.warning("emoji package not available; using fallback regex for emoji counts.")


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _resolve_existing_path(candidates: List[str], what: str) -> Path:
    for c in candidates:
        p = Path(c)
        if p.exists():
            log.info("Using %s: %s", what, p.as_posix())
            return p
    raise FileNotFoundError(
        f"Could not find {what}. Tried: {', '.join(candidates)}"
    )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    df = pd.read_csv(path)
    required_cols = {"text", "sentiment", "source_domain"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def _labels_from_df(df: pd.DataFrame) -> np.ndarray:
    labels = df["sentiment"].map(LABEL_TO_ID)
    if labels.isnull().any():
        bad = sorted(df.loc[labels.isnull(), "sentiment"].unique().tolist())
        raise ValueError(f"Unknown sentiment labels found: {bad}")
    return labels.to_numpy(dtype=np.int64)


def _count_emojis(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    if EMOJI_AVAILABLE:
        return len(emoji.emoji_list(text))
    return len(EMOJI_FALLBACK_PATTERN.findall(text))


def _extract_handcrafted_features(texts: List[str]) -> np.ndarray:
    rows = []
    for text in texts:
        token_count = len(TOKEN_PATTERN.findall(text))
        rows.append(
            [
                _count_emojis(text),
                text.count("!"),
                text.count("?"),
                len(ALL_CAPS_PATTERN.findall(text)),
                token_count,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _load_sparse_or_dense_feature(base_name: str):
    npz_path = Path("data/features") / f"{base_name}.npz"
    npy_path = Path("data/features") / f"{base_name}.npy"
    if npz_path.exists():
        log.info("Loading features from %s", npz_path.as_posix())
        return load_npz(npz_path)
    if npy_path.exists():
        log.info("Loading features from %s", npy_path.as_posix())
        return np.load(npy_path, allow_pickle=False)
    return None


def _build_features_from_text(texts: List[str], vectorizer_path: Path):
    with vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)
    X_tfidf = vectorizer.transform(texts)
    X_hand = _extract_handcrafted_features(texts)
    return X_tfidf, hstack([X_tfidf, csr_matrix(X_hand)], format="csr")


def _pick_logreg_feature_input(
    model, split_name: str, texts: List[str], vectorizer_path: Path
):
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is None:
        raise ValueError("Loaded Logistic Regression model does not expose n_features_in_.")

    precomputed = _load_sparse_or_dense_feature(split_name)
    if precomputed is not None and precomputed.shape[1] == expected_features:
        log.info(
            "Using precomputed '%s' features (shape=%s)",
            split_name,
            tuple(precomputed.shape),
        )
        return precomputed

    X_tfidf, X_combo = _build_features_from_text(texts, vectorizer_path)
    if X_tfidf.shape[1] == expected_features:
        log.info("Model expects TF-IDF-only features for split %s", split_name)
        return X_tfidf
    if X_combo.shape[1] == expected_features:
        log.info("Model expects TF-IDF+handcrafted features for split %s", split_name)
        return X_combo

    raise ValueError(
        "Could not match Logistic Regression model input dimension "
        f"(expected {expected_features}) with available features."
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    class_p, class_r, class_f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    per_class = {}
    for i, label in enumerate(LABEL_ORDER):
        per_class[label] = {
            "precision": float(class_p[i]),
            "recall": float(class_r[i]),
            "f1": float(class_f1[i]),
            "support": int(class_support[i]),
        }

    return {
        "accuracy": float(acc),
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _plot_accuracy_bar(results: Dict, out_path: Path) -> None:
    models = ["logistic_regression", "distilbert"]
    general_vals = [results[m]["general_test"]["accuracy"] for m in models]
    domain_vals = [results[m]["domain_shift_test"]["accuracy"] for m in models]

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, general_vals, width=width, label="General test")
    ax.bar(x + width / 2, domain_vals, width=width, label="Domain shift test")
    ax.set_xticks(x)
    ax.set_xticklabels(["LogReg", "DistilBERT"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy: General vs Domain Shift")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved accuracy comparison plot -> %s", out_path.as_posix())


def _plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved confusion matrix plot -> %s", out_path.as_posix())


def _plot_f1_per_class(results: Dict, out_path: Path) -> None:
    classes = LABEL_ORDER
    x = np.arange(len(classes))
    width = 0.2

    series = [
        (
            "LogReg General",
            [results["logistic_regression"]["general_test"]["per_class"][c]["f1"] for c in classes],
        ),
        (
            "LogReg Domain",
            [results["logistic_regression"]["domain_shift_test"]["per_class"][c]["f1"] for c in classes],
        ),
        (
            "DistilBERT General",
            [results["distilbert"]["general_test"]["per_class"][c]["f1"] for c in classes],
        ),
        (
            "DistilBERT Domain",
            [results["distilbert"]["domain_shift_test"]["per_class"][c]["f1"] for c in classes],
        ),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for i, (name, vals) in enumerate(series):
        ax.bar(x + offsets[i], vals, width=width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-Class F1-score Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved per-class F1 plot -> %s", out_path.as_posix())


def evaluate_logistic_regression(
    logreg_model_path: Path,
    vectorizer_path: Path,
    general_df: pd.DataFrame,
    domain_df: pd.DataFrame,
) -> Dict:
    with logreg_model_path.open("rb") as f:
        model = pickle.load(f)

    y_general = _labels_from_df(general_df)
    y_domain = _labels_from_df(domain_df)

    general_texts = general_df["text"].fillna("").astype(str).tolist()
    domain_texts = domain_df["text"].fillna("").astype(str).tolist()

    X_general = _pick_logreg_feature_input(model, "X_test", general_texts, vectorizer_path)
    X_domain = _pick_logreg_feature_input(model, "X_domain_shift", domain_texts, vectorizer_path)

    y_pred_general = model.predict(X_general)
    y_pred_domain = model.predict(X_domain)

    return {
        "general_test": _compute_metrics(y_general, y_pred_general),
        "domain_shift_test": _compute_metrics(y_domain, y_pred_domain),
    }


def _predict_distilbert(
    model, tokenizer, texts: List[str], labels: np.ndarray, device: torch.device, batch_size: int = 32
) -> np.ndarray:
    ds = TextDataset(texts, labels, tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


def evaluate_distilbert(distilbert_dir: Path, general_df: pd.DataFrame, domain_df: pd.DataFrame) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating DistilBERT on device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
    model = AutoModelForSequenceClassification.from_pretrained(distilbert_dir)
    model.to(device)

    y_general = _labels_from_df(general_df)
    y_domain = _labels_from_df(domain_df)
    general_texts = general_df["text"].fillna("").astype(str).tolist()
    domain_texts = domain_df["text"].fillna("").astype(str).tolist()

    y_pred_general = _predict_distilbert(model, tokenizer, general_texts, y_general, device)
    y_pred_domain = _predict_distilbert(model, tokenizer, domain_texts, y_domain, device)

    return {
        "device": str(device),
        "general_test": _compute_metrics(y_general, y_pred_general),
        "domain_shift_test": _compute_metrics(y_domain, y_pred_domain),
    }


def _print_summary_table(results: Dict) -> None:
    rows = []
    for model_name in ["logistic_regression", "distilbert"]:
        for split_name in ["general_test", "domain_shift_test"]:
            m = results[model_name][split_name]
            rows.append(
                {
                    "model": model_name,
                    "dataset": split_name,
                    "accuracy": round(m["accuracy"], 4),
                    "macro_f1": round(m["macro"]["f1"], 4),
                    "weighted_f1": round(m["weighted"]["f1"], 4),
                    "macro_precision": round(m["macro"]["precision"], 4),
                    "macro_recall": round(m["macro"]["recall"], 4),
                }
            )

    summary_df = pd.DataFrame(rows)
    print("\n=== Evaluation Summary ===")
    print(summary_df.to_string(index=False))


def main() -> None:
    log.info("Starting model evaluation")

    reports_dir = Path("data/reports")
    plots_dir = Path("data/plots")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    general_test_path = Path("data/train/tweets_general_test.csv")
    domain_test_path = Path("data/test_domain_shift/tweets_disaster.csv")
    general_df = _load_csv(general_test_path)
    domain_df = _load_csv(domain_test_path)

    logreg_path = _resolve_existing_path(
        ["models/logreg_model.pkl", "models/logreg_tfidf.pkl"], "Logistic Regression model"
    )
    vectorizer_path = _resolve_existing_path(
        ["models/tfidf_vectorizer.pkl"], "TF-IDF vectorizer"
    )
    distilbert_dir = _resolve_existing_path(
        ["models/distilbert", "models/distilbert_sentiment"], "DistilBERT model directory"
    )

    results = {
        "label_mapping": LABEL_TO_ID,
        "logistic_regression": evaluate_logistic_regression(
            logreg_path, vectorizer_path, general_df, domain_df
        ),
        "distilbert": evaluate_distilbert(distilbert_dir, general_df, domain_df),
    }

    _plot_accuracy_bar(results, plots_dir / "accuracy_comparison_general_vs_domain.png")

    for model_name in ["logistic_regression", "distilbert"]:
        for split_name, split_label in [
            ("general_test", "General Test"),
            ("domain_shift_test", "Domain Shift Test"),
        ]:
            cm = np.array(results[model_name][split_name]["confusion_matrix"])
            file_name = f"confusion_matrix_{model_name}_{split_name}.png"
            title = f"{model_name} - {split_label}"
            _plot_confusion_matrix(cm, title, plots_dir / file_name)

    _plot_f1_per_class(results, plots_dir / "f1_score_per_class_comparison.png")

    results_path = reports_dir / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved evaluation results -> %s", results_path.as_posix())

    _print_summary_table(results)
    log.info("Evaluation complete")


if __name__ == "__main__":
    main()
