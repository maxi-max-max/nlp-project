"""
relabel_disaster.py
-------------------
The disaster domain-shift dataset was originally labelled entirely as
"neutral", which makes domain-shift evaluation degenerate (any model that
just predicts neutral achieves ~69 % accuracy, hiding real performance gaps).

This script re-labels each disaster tweet with a VADER-derived sentiment
(negative / neutral / positive) so that evaluation metrics are meaningful.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
lexicon designed for social-media text and is entirely independent of the
transformer or TF-IDF models we train, so there is no circular evaluation.

Thresholds (standard VADER practice):
  compound >= +0.05  →  positive
  compound <= -0.05  →  negative
  otherwise          →  neutral

Outputs
-------
  data/test_domain_shift/tweets_disaster.csv   (overwritten in-place)
  data/raw/disaster_raw.csv                    (overwritten in-place)
"""

import logging
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def vader_label(text: str, analyser: SentimentIntensityAnalyzer) -> str:
    score = analyser.polarity_scores(str(text))["compound"]
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def relabel(path: Path, analyser: SentimentIntensityAnalyzer) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = df["sentiment"].value_counts().to_dict()
    df["sentiment"] = df["text"].apply(lambda t: vader_label(t, analyser))
    after = df["sentiment"].value_counts().to_dict()
    log.info(
        "%s  |  before: %s  |  after: %s",
        path.name,
        before,
        after,
    )
    df.to_csv(path, index=False)
    return df


def main() -> None:
    analyser = SentimentIntensityAnalyzer()

    paths = [
        Path("data/test_domain_shift/tweets_disaster.csv"),
        Path("data/raw/disaster_raw.csv"),
    ]

    for p in paths:
        if p.exists():
            relabel(p, analyser)
        else:
            log.warning("File not found, skipping: %s", p)

    log.info("Disaster dataset re-labelled with VADER. Re-run evaluate_models.py next.")


if __name__ == "__main__":
    main()
