import json
import logging
import pickle
import re
import hashlib
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
TWITTER_ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

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
    required_cols = {"text", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if "source_domain" not in df.columns:
        inferred_domain = "disaster" if "disaster" in path.name.lower() else "general"
        df["source_domain"] = inferred_domain
        log.warning(
            "Column 'source_domain' missing in %s. Added fallback value '%s' from filename.",
            path.as_posix(),
            inferred_domain,
        )
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_artifact_integrity(artifact_path: Path, reports_dir: Path) -> None:
    manifest_path = reports_dir / "artifact_hashes.json"
    if not manifest_path.exists():
        log.warning("Artifact manifest missing (%s); skipping integrity check.", manifest_path.as_posix())
        return
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected = manifest.get("artifacts", {}).get(artifact_path.as_posix())
    if expected is None:
        log.warning("No manifest entry for %s; skipping integrity check.", artifact_path.as_posix())
        return
    actual_sha = _sha256_file(artifact_path)
    if actual_sha != expected.get("sha256"):
        raise ValueError(f"Integrity check failed for {artifact_path.as_posix()} (SHA256 mismatch).")


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

    observed = np.unique(y_true)
    majority_baseline_accuracy = float(np.max(np.bincount(y_true)) / len(y_true))
    notes = []
    if observed.size < len(LABEL_IDS):
        missing_labels = [ID_TO_LABEL[i] for i in LABEL_IDS if i not in observed]
        notes.append(
            "Single/partial-class split detected; macro/per-class metrics are less representative. "
            f"Missing labels in ground truth: {missing_labels}"
        )

    return {
        "accuracy": float(acc),
        "majority_baseline_accuracy": majority_baseline_accuracy,
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
        "observed_labels": [ID_TO_LABEL[int(i)] for i in observed],
        "notes": notes,
        "confusion_matrix": cm.tolist(),
    }


def _compute_vader_agreement(y_pred: np.ndarray, vader_labels: np.ndarray) -> Dict[str, float]:
    if len(y_pred) != len(vader_labels):
        raise ValueError("Prediction and VADER-label arrays must have identical length.")
    agreement = float(np.mean(y_pred == vader_labels)) if len(y_pred) else 0.0
    return {
        "agreement_accuracy": agreement,
        "agreement_percentage": agreement * 100.0,
    }


def _get_model_keys(results: Dict) -> List[str]:
    return [
        key
        for key, value in results.items()
        if isinstance(value, dict) and "general_test" in value and "domain_shift_test" in value
    ]


def _plot_accuracy_bar(results: Dict, out_path: Path) -> None:
    model_keys = _get_model_keys(results)
    if not model_keys:
        return

    x = np.arange(len(model_keys))
    width = 0.35
    general_vals = [results[m]["general_test"]["accuracy"] for m in model_keys]
    domain_vals = [results[m]["domain_shift_test"]["accuracy"] for m in model_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, general_vals, width=width, label="General test")
    ax.bar(x + width / 2, domain_vals, width=width, label="Domain shift test")
    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Across Test Sets")
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


def _plot_f1_per_class(results: Dict, model_key: str, out_path: Path) -> None:
    classes = LABEL_ORDER
    x = np.arange(len(classes))
    width = 0.35

    general_vals = [
        results[model_key]["general_test"]["per_class"][c]["f1"] for c in classes
    ]
    domain_vals = [
        results[model_key]["domain_shift_test"]["per_class"][c]["f1"] for c in classes
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, general_vals, width=width, label="General test")
    ax.bar(x + width / 2, domain_vals, width=width, label="Domain shift test")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title(f"Per-Class F1-score for {model_key}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("Saved per-class F1 plot -> %s", out_path.as_posix())


def _analyze_vader_distribution(domain_df: pd.DataFrame, reports_dir: Path, plots_dir: Path) -> Dict:
    label_counts = (
        domain_df["sentiment"]
        .astype(str)
        .value_counts(dropna=False)
        .reindex(LABEL_ORDER, fill_value=0)
    )
    total = int(label_counts.sum())
    label_percentages = {
        label: float((count / total) * 100.0 if total else 0.0)
        for label, count in label_counts.items()
    }

    analysis = {
        "total_samples": total,
        "label_counts": {label: int(count) for label, count in label_counts.items()},
        "label_percentages": label_percentages,
    }

    if "vader_compound" not in domain_df.columns:
        analysis["warning"] = (
            "Column 'vader_compound' is missing in domain-shift CSV. "
            "Run relabel_disaster.py to generate score-distribution diagnostics."
        )
        log.warning(analysis["warning"])
    else:
        compounds = pd.to_numeric(domain_df["vader_compound"], errors="coerce").dropna().to_numpy()
        if compounds.size == 0:
            analysis["warning"] = "Column 'vader_compound' exists but contains no numeric values."
            log.warning(analysis["warning"])
        else:
            neutral_mask = np.logical_and(compounds > -0.05, compounds < 0.05)
            analysis["compound_stats"] = {
                "count": int(compounds.size),
                "mean": float(np.mean(compounds)),
                "std": float(np.std(compounds)),
                "min": float(np.min(compounds)),
                "max": float(np.max(compounds)),
                "q25": float(np.quantile(compounds, 0.25)),
                "median": float(np.quantile(compounds, 0.50)),
                "q75": float(np.quantile(compounds, 0.75)),
                "neutral_band_percentage": float(np.mean(neutral_mask) * 100.0),
            }

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(compounds, bins=60, color="#1f77b4", alpha=0.85, edgecolor="white")
            ax.axvline(-0.05, color="red", linestyle="--", linewidth=1.5, label="VADER negative threshold")
            ax.axvline(0.05, color="green", linestyle="--", linewidth=1.5, label="VADER positive threshold")
            ax.set_title("Disaster Set VADER Compound Score Distribution")
            ax.set_xlabel("VADER compound score")
            ax.set_ylabel("Tweet count")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            out_plot = plots_dir / "vader_compound_distribution_disaster.png"
            fig.savefig(out_plot, dpi=200)
            plt.close(fig)
            log.info("Saved VADER compound distribution plot -> %s", out_plot.as_posix())
            analysis["compound_distribution_plot"] = out_plot.as_posix()

    out_json = reports_dir / "domain_shift_vader_analysis.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    log.info("Saved VADER label/compound analysis -> %s", out_json.as_posix())
    return analysis


def evaluate_logistic_regression(
    logreg_model_path: Path,
    vectorizer_path: Path,
    reports_dir: Path,
    general_df: pd.DataFrame,
    domain_df: pd.DataFrame,
) -> Dict:
    _verify_artifact_integrity(logreg_model_path, reports_dir)
    _verify_artifact_integrity(vectorizer_path, reports_dir)
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

    domain_metrics = _compute_metrics(y_domain, y_pred_domain)
    domain_metrics["label_source"] = "VADER (pseudo-labels, not human-annotated)"
    domain_metrics["vader_agreement"] = _compute_vader_agreement(y_pred_domain, y_domain)

    return {
        "general_test": _compute_metrics(y_general, y_pred_general),
        "domain_shift_test": domain_metrics,
    }


def _predict_twitter_roberta(
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


def evaluate_twitter_roberta(model_dir: Path, general_df: pd.DataFrame, domain_df: pd.DataFrame) -> Dict:
    reports_dir = Path("data/reports")
    for f in [
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
        model_dir / "vocab.json",
        model_dir / "merges.txt",
    ]:
        if f.exists():
            _verify_artifact_integrity(f, reports_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating Twitter-RoBERTa on device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    y_general = _labels_from_df(general_df)
    y_domain = _labels_from_df(domain_df)
    general_texts = general_df["text"].fillna("").astype(str).tolist()
    domain_texts = domain_df["text"].fillna("").astype(str).tolist()

    y_pred_general = _predict_twitter_roberta(model, tokenizer, general_texts, y_general, device)
    y_pred_domain = _predict_twitter_roberta(model, tokenizer, domain_texts, y_domain, device)

    domain_metrics = _compute_metrics(y_domain, y_pred_domain)
    domain_metrics["label_source"] = "VADER (pseudo-labels, not human-annotated)"
    domain_metrics["vader_agreement"] = _compute_vader_agreement(y_pred_domain, y_domain)

    return {
        "device": str(device),
        "general_test": _compute_metrics(y_general, y_pred_general),
        "domain_shift_test": domain_metrics,
    }


def evaluate_twitter_roberta_zero_shot(general_df: pd.DataFrame, domain_df: pd.DataFrame) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating zero-shot Twitter-RoBERTa (%s) on device: %s", TWITTER_ROBERTA_MODEL, device)

    tokenizer = AutoTokenizer.from_pretrained(TWITTER_ROBERTA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(TWITTER_ROBERTA_MODEL)
    model.to(device)

    y_general = _labels_from_df(general_df)
    y_domain = _labels_from_df(domain_df)
    general_texts = general_df["text"].fillna("").astype(str).tolist()
    domain_texts = domain_df["text"].fillna("").astype(str).tolist()

    y_pred_general = _predict_twitter_roberta(model, tokenizer, general_texts, y_general, device)
    y_pred_domain = _predict_twitter_roberta(model, tokenizer, domain_texts, y_domain, device)

    domain_metrics = _compute_metrics(y_domain, y_pred_domain)
    domain_metrics["label_source"] = "VADER (pseudo-labels, not human-annotated)"
    domain_metrics["vader_agreement"] = _compute_vader_agreement(y_pred_domain, y_domain)

    return {
        "model_name": TWITTER_ROBERTA_MODEL,
        "setup": "zero_shot_pretrained",
        "device": str(device),
        "general_test": _compute_metrics(y_general, y_pred_general),
        "domain_shift_test": domain_metrics,
    }


def _print_summary_table(results: Dict) -> None:
    rows = []
    model_keys = _get_model_keys(results)
    for model_key in model_keys:
        for split_name in ["general_test", "domain_shift_test"]:
            m = results[model_key][split_name]
            rows.append(
                {
                    "model": model_key,
                    "dataset": split_name,
                    "accuracy": round(m["accuracy"], 4),
                    "macro_f1": round(m["macro"]["f1"], 4),
                    "weighted_f1": round(m["weighted"]["f1"], 4),
                    "majority_baseline_accuracy": round(m["majority_baseline_accuracy"], 4),
                    "macro_precision": round(m["macro"]["precision"], 4),
                    "macro_recall": round(m["macro"]["recall"], 4),
                }
            )

    summary_df = pd.DataFrame(rows)
    print("\n=== Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(
        "\nWARNING: Domain shift metrics are evaluated against VADER-derived pseudo-labels, "
        "not human annotations."
    )


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
        ["models/logreg_model.pkl", "models/logreg_tfidf.pkl"],
        "Logistic Regression model"
    )
    vectorizer_path = _resolve_existing_path(
        ["models/tfidf_vectorizer.pkl"],
        "TF-IDF vectorizer"
    )

    results = {
        "label_mapping": LABEL_TO_ID,
        "domain_shift_warning": (
            "Domain shift metrics are evaluated against VADER-derived pseudo-labels, "
            "not human annotations."
        ),
        "logistic_regression": evaluate_logistic_regression(
            logreg_path, vectorizer_path, reports_dir, general_df, domain_df
        ),
    }

    results["twitter_roberta_zero_shot"] = evaluate_twitter_roberta_zero_shot(general_df, domain_df)
    twitter_roberta_dir = Path("models/twitter_roberta_sentiment")
    if twitter_roberta_dir.exists():
        results["twitter_roberta"] = evaluate_twitter_roberta(
            twitter_roberta_dir, general_df, domain_df
        )
    else:
        log.warning(
            "Twitter-RoBERTa directory not found at %s; skipping transformer evaluation.",
            twitter_roberta_dir.as_posix(),
        )

    results["domain_shift_vader_analysis"] = _analyze_vader_distribution(
        domain_df=domain_df,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
    )

    _plot_accuracy_bar(results, plots_dir / "accuracy_comparison_general_vs_domain.png")

    model_keys = _get_model_keys(results)
    for model_key in model_keys:
        for split_name, split_label in [
            ("general_test", "General Test"),
            ("domain_shift_test", "Domain Shift Test"),
        ]:
            cm = np.array(results[model_key][split_name]["confusion_matrix"])
            file_name = f"confusion_matrix_{model_key}_{split_name}.png"
            title = f"{model_key} - {split_label}"
            _plot_confusion_matrix(cm, title, plots_dir / file_name)

        _plot_f1_per_class(
            results,
            model_key=model_key,
            out_path=plots_dir / f"f1_score_per_class_{model_key}.png",
        )

    results_path = reports_dir / "evaluation_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved evaluation results -> %s", results_path.as_posix())

    _print_summary_table(results)
    log.info("Evaluation complete")


if __name__ == "__main__":
    main()
