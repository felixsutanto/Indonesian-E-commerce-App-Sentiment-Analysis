# E-commerce Review Sentiment Analysis

Transformer-based sentiment analysis for e-commerce app reviews (e.g., Tokopedia, Shopee), with a full pipeline for data validation, preprocessing, model training, evaluation, and logging.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Tech Stack](#tech-stack)  
4. [Project Structure](#project-structure)  
5. [Getting Started](#getting-started)  
6. [Data Description](#data-description)  
7. [Data Validation & Quality Reports](#data-validation--quality-reports)  
8. [Model Training & Evaluation](#model-training--evaluation)  
9. [Logging](#logging)  
10. [Testing](#testing)  
11. [Possible Extensions](#possible-extensions)  
12. [License](#license)

---

## Project Overview

This project builds a sentiment analysis system for e-commerce product/app reviews. The goal is to classify user reviews into **negative**, **neutral**, and **positive** sentiment, with a focus on:

- Real-world e-commerce platforms (e.g., Tokopedia, Shopee).
- Multilingual text (e.g., Bahasa Indonesia, English) using a **multilingual transformer** model.
- A robust data pipeline that includes:
  - Data validation and quality checks.
  - Text preprocessing.
  - Transformer fine-tuning (Hugging Face).
  - Comprehensive evaluation and logging.

The project is designed to be a practical, end-to-end example that can be adapted to other domains and datasets.

---

## Key Features

- **Transformer-based classifier**
  - Uses `distilbert-base-multilingual-cased` for sequence classification.
  - Fine-tuned using Hugging Face `Trainer` with early stopping.
  - Supports **3 classes**: `negative`, `neutral`, `positive`.

- **Data validation**
  - `DataValidator` class for:
    - Schema checks and required columns.
    - Missing value and data type analysis.
    - Score validity checks (1–5 rating).
    - Text length statistics and anomalies.
    - Class imbalance detection on processed data.
  - Generates **visual data quality reports** (histograms, distributions).

- **Preprocessing pipeline**
  - Dedicated preprocessing script (e.g., `ReviewPreprocessor`) to:
    - Clean raw review text.
    - Map review scores to sentiment labels.
    - Compute token counts.
    - Output a clean, model-ready dataset.

- **Logging**
  - Centralized logging configuration with Windows-friendly UTF-8 handling.
  - `ModelLogger` for:
    - Dataset statistics.
    - Training progress.
    - Model performance metrics.

- **Evaluation**
  - Accuracy, precision, recall, F1 (weighted).
  - Confusion matrix plotted and saved to disk.
  - Text classification report.

- **Tests**
  - Simple smoke tests to verify imports and preprocessing behave as expected.

---

## Tech Stack

- **Language**
  - Python (3.9+ recommended)

- **Core Libraries**
  - [PyTorch](https://pytorch.org/) – model backbone
  - [Transformers](https://github.com/huggingface/transformers) – DistilBERT and Trainer API
  - [Datasets](https://github.com/huggingface/datasets) – dataset wrapping
  - [scikit-learn](https://scikit-learn.org/) – metrics, train/test split
  - [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) – data manipulation
  - [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) – visualization

- **Utilities**
  - Standard Python `logging`
  - Custom logging and data validation utilities

---

## Project Structure

A typical layout for this repository:

```text
.
├── scripts/
│   ├── transformer_model.py      # TransformerClassifier: training, evaluation, prediction
│   └── datapreprocessor.py       # ReviewPreprocessor: raw → processed data (not shown here)
│
├── utils/
│   ├── data_validation.py        # DataValidator: validation & data quality reporting
│   └── logging_config.py         # Logging setup & ModelLogger
│
├── data/
│   ├── raw/                      # Raw scraped reviews (CSV)
│   └── processed/                # Cleaned dataset for modeling
│       └── processed_reviews.csv
│
├── models/                       # Saved transformer checkpoints
│   └── distilbert/               # Default output dir for DistilBERT fine-tuning
│
├── results/                      # Evaluation artifacts (e.g., confusion matrix)
│   └── transformer_confusion_matrix.png
│
├── logs/                         # Log files for training & evaluation
│
├── tests/
│   └── test_fixes.py             # Smoke test for preprocessing pipeline
│
├── requirements.txt
└── README.md
```

Adjust the paths in your scripts if your structure differs.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```


### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure `torch`, `transformers`, `datasets`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn` are included in `requirements.txt`.

---

## Data Description

### Raw data

The **raw review data** is expected to be a CSV file with at least the following columns:

- `content` – Original review text written by the user.
- `score` – Numerical rating, typically from **1 to 5**.
- `appname` – Source platform (e.g., `tokopedia`, `shopee`).

Example (simplified):

```csv
content,score,appname
"Aplikasi sangat bagus!",5,tokopedia
"App jelek banget",1,shopee
"Biasa aja",3,tokopedia
```

Place this file under:

```text
data/raw/raw_reviews.csv
```

(or adjust paths in your scripts accordingly).

### Processed data

The **processed dataset** used for training the transformer is expected to contain:

- `cleaned_content` – Preprocessed text (tokenized/normalized).
- `sentiment` – Final sentiment label:
    - `negative`
    - `neutral`
    - `positive`
- `tokencount` – (Optional) number of tokens in each processed review.
- `appname` – (Optional) platform label, propagated from raw data.

This is typically saved as:

```text
data/processed/processed_reviews.csv
```

The `TransformerClassifier` uses this file by default in `transformer_model.py`.

---

## Data Validation \& Quality Reports

Data quality is handled by `DataValidator` in `utils/data_validation.py`.

### Key checks

For **raw data**:

- Total records.
- Missing values per column (count and percentage).
- Column data types.
- Unique values for categorical columns (e.g., `appname`, `score`).
- Score validity (e.g., ensuring `score` is between 1 and 5).
- Review text length statistics:
    - min, max, mean, median.
    - number of **very short** and **very long** reviews.

For **processed data**:

- Required columns: `cleaned_content`, `sentiment`, `tokencount`.
- Sentiment distribution and **class imbalance** detection.
- Empty content after cleaning.
- Token count distribution and extreme cases (e.g., very short processed reviews).


### Usage example

```python
from utils.data_validation import DataValidator
import pandas as pd

raw_df = pd.read_csv("data/raw/raw_reviews.csv")
processed_df = pd.read_csv("data/processed/processed_reviews.csv")

validator = DataValidator()

raw_results = validator.validate_raw_data(raw_df)
processed_results = validator.validate_processed_data(processed_df)

# Optional: generate a visual report
validator.create_data_quality_report(
    dfraw=raw_df,
    dfprocessed=processed_df,
    savepath="results/"  # saves dataqualityreport.png
)

# Print a human-readable summary
validator.print_validation_summary()
```

This will generate plots such as:

- Raw score distribution.
- Raw text length distribution.
- Missing values per column.
- Processed sentiment distribution.
- Token count distribution.
- App/platform distribution.

---

## Model Training \& Evaluation

The main transformer logic is implemented in `scripts/transformer_model.py` via the `TransformerClassifier` class.

### Label mapping

Sentiment labels are mapped internally as:

- `negative` → `0`
- `neutral` → `1`
- `positive` → `2`

This mapping is used consistently in the model configuration and predictions.

### Training pipeline

Key steps in `TransformerClassifier.train`:

1. **Tokenization**
Uses `AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")`
    - `truncation=True`
    - `padding="max_length"`
    - `max_length=128`
2. **Dataset preparation**
Wraps data using `datasets.Dataset` and applies tokenization with `.map(...)`.
3. **Model setup**
Loads `AutoModelForSequenceClassification` with:
    - `num_labels=3`
    - `id2label` and `label2id` mappings.
4. **TrainingArguments**
Example configuration:
    - `num_train_epochs=3`
    - `per_device_train_batch_size=16`
    - `per_device_eval_batch_size=16`
    - `warmup_steps=500`
    - `weight_decay=0.01`
    - `evaluation_strategy="steps"`
    - `eval_steps=500`
    - `save_strategy="steps"`
    - `save_steps=500`
    - `load_best_model_at_end=True`
    - `metric_for_best_model="f1"`
    - `report_to=None` (disables external logging backends like WandB)
    - `seed=42`
5. **Callbacks**
    - Uses `EarlyStoppingCallback` with a patience of 3 evaluation steps.
6. **Model saving**
    - Saves the fine-tuned model and tokenizer to `output_dir` (default: `models/distilbert`).

### Running training

By default (as implemented in the `__main__` block of `transformer_model.py`), you can:

```bash
python scripts/transformer_model.py
```

This will typically:

- Load `data/processed/processed_reviews.csv`.
- Split into train/validation/test sets.
- Fine-tune DistilBERT.
- Evaluate on the test set.
- Save:
    - Model weights to `models/distilbert/`.
    - Confusion matrix plot to `results/transformer_confusion_matrix.png`.

Ensure `data/processed/processed_reviews.csv`, `models/`, and `results/` directories exist (they can be created manually if needed).

### Evaluation and metrics

The `evaluate` method computes:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

It also:

- Prints a detailed `classification_report`.
- Generates and saves a **confusion matrix** plot with classes:
    - `negative`, `neutral`, `positive`.

---

## Logging

Logging is centralized in `utils/logging_config.py`.

### Global logging setup

The `set_up_logging` (or similarly named) function:

- Configures the root logger.
- Uses a consistent, emoji-free log format compatible with Windows terminals.
- Tries to enforce UTF-8 encoding for console logs, with a fallback if not available.
- Optionally writes logs to a specified file (e.g., `logs/sentiment_analysis.log`).

Example usage:

```python
from utils.logging_config import setup_logging

logger = setup_logging(loglevel="INFO", logfile="logs/sentiment_analysis.log")
logger.info("Logging system initialized successfully")
```


### Model-specific logging

`ModelLogger` is a helper class for model-related logging:

- `log_data_info(df, stage)`
Logs:
    - Shape of the dataset.
    - Columns.
    - Sentiment distribution (if `sentiment` column present).
- `log_model_performance(model_name, metrics)`
Logs:
    - Accuracy.
    - Weighted precision, recall, F1-score.
- `log_training_progress(epoch, loss, metrics=None)`
Logs epoch-wise training loss and metrics.

Example:

```python
from utils.logging_config import ModelLogger

model_logger = ModelLogger(name="distilbert", logfile="logs/model_training.log")
model_logger.log_training_progress(epoch=1, loss=0.45, metrics={"f1": 0.82, "accuracy": 0.84})
```


---

## Testing

A simple smoke test script is provided in `tests/test_fixes.py` to verify that:

- Imports work correctly (e.g., `ReviewPreprocessor`).
- Basic preprocessing runs without errors on sample data.
- Sentiment distribution on the sample output looks reasonable.

Run:

```bash
python tests/test_fixes.py
```

If everything is configured correctly, the script prints messages indicating successful preprocessing and basic statistics about the processed data.

---

## Possible Extensions

Some ideas to extend or customize this project:

- **More granular sentiment**
    - Add labels like `very negative`, `very positive`.
- **Domain adaptation**
    - Fine-tune with reviews from other platforms or product categories.
- **Hyperparameter optimization**
    - Integrate Optuna or Ray Tune for automatic search.
- **Model comparison**
    - Add additional models (e.g., BiLSTM, other BERT variants) and compare performance.
- **Deployment**
    - Wrap the model into a REST API using FastAPI or Flask.
    - Create a simple web UI to input a review and get real-time sentiment predictions.

---

## License

Add your chosen license here, for example:

```text
MIT License

Copyright (c) 2026 <Your Name>
```

Or reference a `LICENSE` file if you include one in the repository.

---
```
<span style="display:none">[^1][^2][^3][^4]</span>

<div align="center">⁂</div>

[^1]: transformer_model.py
[^2]: data_validation.py
[^3]: logging_config.py
[^4]: test_fixes.py```

