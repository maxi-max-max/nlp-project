import json
import logging
import pickle
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_extraction.log"),
    ],
)
log = logging.getLogger(__name__)


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
    log.warning("emoji package not available; using fallback regex for emoji count.")


def _count_emojis(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    if EMOJI_AVAILABLE:
        return len(emoji.emoji_list(text))
    return len(EMOJI_FALLBACK_PATTERN.findall(text))


def _extract_handcrafted_features(text_series: pd.Series) -> np.ndarray:
    rows = []
    for text in text_series.fillna(""):
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


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    required = {"text", "sentiment", "source_domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in '{path}': {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path)
    _validate_columns(df, path)
    log.info("Loaded %s with %d rows", path.as_posix(), len(df))
    return df


def _feature_stats(arr: np.ndarray) -> dict:
    feature_names = [
        "emoji_count",
        "exclamation_count",
        "question_count",
        "all_caps_word_count",
        "text_length_tokens",
    ]
    stats = {}
    for idx, name in enumerate(feature_names):
        col = arr[:, idx]
        stats[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return stats


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_artifact_manifest(reports_dir: Path, artifacts: list[Path]) -> None:
    manifest_path = reports_dir / "artifact_hashes.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"updated_at": None, "artifacts": {}}

    entries = manifest.setdefault("artifacts", {})
    for artifact in artifacts:
        entries[artifact.as_posix()] = {
            "sha256": _sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Updated artifact hash manifest -> %s", manifest_path.as_posix())


def main() -> None:
    log.info("Starting feature extraction pipeline")

    train_path = Path("data/train/tweets_general_train.csv")
    val_path = Path("data/train/tweets_general_val.csv")
    test_path = Path("data/train/tweets_general_test.csv")
    domain_path = Path("data/test_domain_shift/tweets_disaster.csv")

    models_dir = Path("models")
    features_dir = Path("data/features")
    reports_dir = Path("data/reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split(train_path)
    val_df = _load_split(val_path)
    test_df = _load_split(test_path)
    domain_df = _load_split(domain_path)

    log.info("Fitting TF-IDF vectorizer on training text")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        lowercase=False,
    )
    X_train_tfidf = vectorizer.fit_transform(train_df["text"].fillna(""))
    X_val_tfidf = vectorizer.transform(val_df["text"].fillna(""))
    X_test_tfidf = vectorizer.transform(test_df["text"].fillna(""))
    X_domain_tfidf = vectorizer.transform(domain_df["text"].fillna(""))
    log.info(
        "TF-IDF matrices built (vocab size=%d)",
        len(vectorizer.get_feature_names_out()),
    )

    log.info("Extracting handcrafted tweet-level features")
    X_train_hand = _extract_handcrafted_features(train_df["text"])
    X_val_hand = _extract_handcrafted_features(val_df["text"])
    X_test_hand = _extract_handcrafted_features(test_df["text"])
    X_domain_hand = _extract_handcrafted_features(domain_df["text"])

    log.info("Combining TF-IDF and handcrafted features")
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_hand)], format="csr")
    X_val = hstack([X_val_tfidf, csr_matrix(X_val_hand)], format="csr")
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_hand)], format="csr")
    X_domain_shift = hstack([X_domain_tfidf, csr_matrix(X_domain_hand)], format="csr")

    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    with vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)
    log.info("Saved TF-IDF vectorizer to %s", vectorizer_path.as_posix())
    _update_artifact_manifest(reports_dir, [vectorizer_path])

    save_npz(features_dir / "X_train.npz", X_train)
    save_npz(features_dir / "X_val.npz", X_val)
    save_npz(features_dir / "X_test.npz", X_test)
    save_npz(features_dir / "X_domain_shift.npz", X_domain_shift)
    log.info("Saved feature matrices to %s", features_dir.as_posix())

    report = {
        "vocabulary_size": int(len(vectorizer.get_feature_names_out())),
        "tfidf_config": {
            "ngram_range": [1, 2],
            "max_features": 50000,
        },
        "splits": {
            "train": {
                "rows": int(X_train.shape[0]),
                "cols": int(X_train.shape[1]),
                "nnz": int(X_train.nnz),
                "density": float(X_train.nnz / (X_train.shape[0] * X_train.shape[1])),
                "handcrafted_stats": _feature_stats(X_train_hand),
            },
            "val": {
                "rows": int(X_val.shape[0]),
                "cols": int(X_val.shape[1]),
                "nnz": int(X_val.nnz),
                "density": float(X_val.nnz / (X_val.shape[0] * X_val.shape[1])),
                "handcrafted_stats": _feature_stats(X_val_hand),
            },
            "test": {
                "rows": int(X_test.shape[0]),
                "cols": int(X_test.shape[1]),
                "nnz": int(X_test.nnz),
                "density": float(X_test.nnz / (X_test.shape[0] * X_test.shape[1])),
                "handcrafted_stats": _feature_stats(X_test_hand),
            },
            "domain_shift": {
                "rows": int(X_domain_shift.shape[0]),
                "cols": int(X_domain_shift.shape[1]),
                "nnz": int(X_domain_shift.nnz),
                "density": float(
                    X_domain_shift.nnz / (X_domain_shift.shape[0] * X_domain_shift.shape[1])
                ),
                "handcrafted_stats": _feature_stats(X_domain_hand),
            },
        },
    }

    report_path = reports_dir / "feature_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("Saved feature report to %s", report_path.as_posix())
    log.info("Feature extraction pipeline complete")


if __name__ == "__main__":
    main()
