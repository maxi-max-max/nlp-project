# Sentiment Analysis on Social Media — NLP Course Project (Group 11)

## Project Overview
This project studies how well Transformer-based models perform for sentiment analysis on social media text. We compare a traditional machine learning baseline (Logistic Regression with TF-IDF features) against a fine-tuned Twitter-RoBERTa model, and evaluate robustness under domain shift (training on general tweets, testing on disaster-related tweets).

## Team
Leena El Earq, Blanca Valdes, Jaime Paz, Maximilliano Martin, Omar El Haj  
IE University — NLP Course Project

## Project Structure
```
nlp-project/
├── feature_extraction.py    # TF-IDF + handcrafted feature extraction
├── train_models.py          # Train Logistic Regression + Twitter-RoBERTa
├── evaluate_models.py       # Evaluate both models, generate plots & metrics
├── error_analysis.py        # Categorise and analyse misclassified examples
├── relabel_disaster.py      # Re-label disaster tweets with VADER for valid domain-shift eval
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw/                 # Original downloaded datasets
│   ├── train/               # Cleaned train/val/test CSV splits
│   ├── test_domain_shift/   # Disaster tweets for domain shift evaluation
│   ├── features/            # Saved TF-IDF feature matrices (.npz)
│   ├── reports/             # JSON evaluation results & statistics
│   └── plots/               # Generated figures (confusion matrices, bar charts)
└── models/                  # Saved model weights and tokenizers
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Feature extraction
```bash
python feature_extraction.py
```
Fits a TF-IDF vectorizer on training data, extracts handcrafted features (emoji count, punctuation counts, caps words, token length), and saves sparse matrices to `data/features/`.

### 3. Model training
```bash
python train_models.py
```
Trains:
- **Logistic Regression** on TF-IDF + handcrafted features
- **Twitter-RoBERTa** (fine-tuned from `cardiffnlp/twitter-roberta-base-sentiment-latest`)

### 4. Re-label disaster dataset (required before evaluation)
```bash
python relabel_disaster.py
```
The original disaster dataset contains only `neutral` labels (it was a topic-classification dataset, not a sentiment dataset). This script uses VADER — a rule-based sentiment lexicon for social media, entirely independent of our trained models — to assign meaningful positive/negative/neutral labels. This is required to make domain-shift evaluation valid.

### 5. Evaluation
```bash
python evaluate_models.py
```
Evaluates both models on general test set and domain-shift (disaster) test set. Outputs metrics to `data/reports/evaluation_results.json` and plots to `data/plots/`.

### 6. Error analysis
```bash
python error_analysis.py
```
Categorises misclassified examples into error types (negation, sarcasm, emoji conflict, mixed sentiment, lack of context). Saves results to `data/reports/error_analysis.json`.

## Datasets
- **General tweets**: TweetEval sentiment dataset (3 classes: positive, negative, neutral)
- **Domain shift**: Disaster-related tweets for out-of-domain evaluation

## Models
| Model | Description |
|-------|-------------|
| Logistic Regression | TF-IDF (unigrams + bigrams, 50K features) + 5 handcrafted features, L2 regularisation |
| Twitter-RoBERTa | Fine-tuned `cardiffnlp/twitter-roberta-base-sentiment-latest` (pretrained for tweet sentiment) |

## Evaluation Metrics
- Accuracy, Macro/Weighted Precision, Recall, F1-score
- Per-class F1 scores
- Confusion matrices
- Error category analysis

## Requirements
- Python 3.10+
- See `requirements.txt` for full dependency list
- GPU optional (CPU training supported)
