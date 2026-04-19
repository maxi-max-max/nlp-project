import logging
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("relabel_disaster.log"),
    ],
)
log = logging.getLogger(__name__)


def _score_to_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _relabel_file(path: Path, analyzer: SentimentIntensityAnalyzer) -> None:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"'text' column missing in {path}")

    if "source_domain" not in df.columns and "disaster" in path.name.lower():
        df["source_domain"] = "disaster"
        log.warning(
            "Column 'source_domain' was missing in %s; defaulted to 'disaster'.",
            path.as_posix(),
        )

    text_series = df["text"].fillna("").astype(str)
    vader_compound = text_series.apply(lambda x: float(analyzer.polarity_scores(x)["compound"]))
    df["vader_compound"] = vader_compound
    df["sentiment"] = vader_compound.apply(_score_to_label)
    df.to_csv(path, index=False)

    class_distribution = df["sentiment"].value_counts(dropna=False).to_dict()
    log.info("Re-labeled with VADER and saved to %s", path.as_posix())
    log.info("Saved transparency column 'vader_compound' with raw VADER scores.")
    log.info("Class distribution after relabeling: %s", class_distribution)


def main() -> None:
    analyzer = SentimentIntensityAnalyzer()
    candidate_paths = [
        Path("data/test_domain_shift/tweets_disaster.csv"),
        Path("data/raw/disaster_raw.csv"),
    ]
    existing_paths = [path for path in candidate_paths if path.exists()]
    if not existing_paths:
        raise FileNotFoundError(
            "Missing disaster dataset files. Expected at least one of: "
            "data/test_domain_shift/tweets_disaster.csv, data/raw/disaster_raw.csv"
        )

    for path in existing_paths:
        _relabel_file(path, analyzer)


if __name__ == "__main__":
    main()
