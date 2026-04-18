import json
import logging
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
TWITTER_ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _ensure_dirs() -> Tuple[Path, Path]:
    models_dir = Path("models")
    reports_dir = Path("data/reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, reports_dir


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_artifact_manifest(reports_dir: Path, artifacts: List[Path]) -> None:
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

    manifest["updated_at"] = pd.Timestamp.now("UTC").isoformat()
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Updated artifact hash manifest -> %s", manifest_path.as_posix())


def _load_labels(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing label source CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "sentiment" not in df.columns:
        raise ValueError(f"'sentiment' column missing in {csv_path}")
    labels = df["sentiment"].map(LABEL_TO_ID)
    if labels.isnull().any():
        unknown = sorted(df.loc[labels.isnull(), "sentiment"].unique().tolist())
        raise ValueError(f"Unknown labels in {csv_path}: {unknown}")
    return labels.to_numpy(dtype=np.int64)


def _load_texts(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing text source CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError(f"'text' column missing in {csv_path}")
    return df["text"].fillna("").astype(str).tolist()


def _load_feature_matrix(base_name: str):
    npz_path = Path("data/features") / f"{base_name}.npz"
    npy_path = Path("data/features") / f"{base_name}.npy"

    if npz_path.exists():
        log.info("Loading sparse features from %s", npz_path.as_posix())
        return load_npz(npz_path)
    if npy_path.exists():
        log.info("Loading dense features from %s", npy_path.as_posix())
        return np.load(npy_path, allow_pickle=False)
    raise FileNotFoundError(
        f"Could not find feature matrix for {base_name}. Expected {npz_path} or {npy_path}."
    )


class SentimentTextDataset(Dataset):
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


def train_logistic_regression(
    X_train, y_train: np.ndarray, X_val, y_val: np.ndarray, models_dir: Path, reports_dir: Path
) -> Dict:
    log.info("Training Logistic Regression baseline on saved TF-IDF features")
    # sklearn >= 1.8: `multi_class` was removed; multiclass is handled automatically.
    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    log.info("Logistic Regression validation accuracy: %.4f", val_acc)

    out_path = models_dir / "logreg_tfidf.pkl"
    with out_path.open("wb") as f:
        pickle.dump(model, f)
    log.info("Saved Logistic Regression model to %s", out_path.as_posix())
    _update_artifact_manifest(reports_dir, [out_path])

    return {
        "model_name": "tfidf_logistic_regression",
        "validation_accuracy": float(val_acc),
        "model_path": out_path.as_posix(),
    }


def evaluate_twitter_roberta(
    model: AutoModelForSequenceClassification, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    preds_all = []
    labels_all = []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds_all.append(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(labels, preds)
    return {"loss": float(avg_loss), "accuracy": float(acc)}


def train_twitter_roberta(
    train_texts: List[str],
    y_train: np.ndarray,
    val_texts: List[str],
    y_val: np.ndarray,
    models_dir: Path,
    reports_dir: Path,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    epochs: int = 3,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training Twitter-RoBERTa on device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(TWITTER_ROBERTA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        TWITTER_ROBERTA_MODEL,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,  # suppress UNEXPECTED pooler key warnings
    )
    model.to(device)

    train_ds = SentimentTextDataset(train_texts, y_train, tokenizer)
    val_ds = SentimentTextDataset(val_texts, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_training_steps)),
        num_training_steps=total_training_steps,
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_twitter_roberta(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        log.info(
            "Twitter-RoBERTa epoch %d/%d - train_loss: %.4f - val_loss: %.4f - val_acc: %.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_acc,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }
        )

    out_dir = models_dir / "twitter_roberta_sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info("Saved Twitter-RoBERTa model to %s", out_dir.as_posix())
    artifact_files = [
        out_dir / "config.json",
        out_dir / "model.safetensors",
        out_dir / "tokenizer.json",
        out_dir / "tokenizer_config.json",
        out_dir / "vocab.json",
        out_dir / "merges.txt",
    ]
    _update_artifact_manifest(reports_dir, [p for p in artifact_files if p.exists()])

    return {
        "model_name": TWITTER_ROBERTA_MODEL,
        "device": str(device),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "model_path": out_dir.as_posix(),
        "history": history,
    }


def main() -> None:
    log.info("Starting model training pipeline")
    models_dir, reports_dir = _ensure_dirs()

    train_csv = Path("data/train/tweets_general_train.csv")
    val_csv = Path("data/train/tweets_general_val.csv")

    y_train = _load_labels(train_csv)
    y_val = _load_labels(val_csv)
    X_train = _load_feature_matrix("X_train")
    X_val = _load_feature_matrix("X_val")

    if X_train.shape[0] != len(y_train):
        raise ValueError(f"X_train rows ({X_train.shape[0]}) != y_train size ({len(y_train)})")
    if X_val.shape[0] != len(y_val):
        raise ValueError(f"X_val rows ({X_val.shape[0]}) != y_val size ({len(y_val)})")

    logreg_result = train_logistic_regression(
        X_train, y_train, X_val, y_val, models_dir, reports_dir
    )

    train_texts = _load_texts(train_csv)
    val_texts = _load_texts(val_csv)
    # Fine-tune on a manageable subset to keep CPU training time reasonable
    # (the pretrained model is already specialised for tweet sentiment)
    max_train = 5000
    max_val = 1000
    if len(train_texts) > max_train:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_texts), max_train, replace=False)
        train_texts_sub = [train_texts[i] for i in idx]
        y_train_sub = y_train[idx]
    else:
        train_texts_sub, y_train_sub = train_texts, y_train

    if len(val_texts) > max_val:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(val_texts), max_val, replace=False)
        val_texts_sub = [val_texts[i] for i in idx]
        y_val_sub = y_val[idx]
    else:
        val_texts_sub, y_val_sub = val_texts, y_val

    twitter_roberta_result = train_twitter_roberta(
        train_texts=train_texts_sub,
        y_train=y_train_sub,
        val_texts=val_texts_sub,
        y_val=y_val_sub,
        models_dir=models_dir,
        reports_dir=reports_dir,
        batch_size=16,
        learning_rate=2e-5,
        epochs=1,
    )

    history_payload = {
        "label_mapping": LABEL_TO_ID,
        "logistic_regression": logreg_result,
        "twitter_roberta": twitter_roberta_result,
    }
    history_path = reports_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    log.info("Saved training history to %s", history_path.as_posix())
    log.info("Model training pipeline complete")


if __name__ == "__main__":
    main()
