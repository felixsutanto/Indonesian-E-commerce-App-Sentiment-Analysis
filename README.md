# E-commerce Review Sentiment Analysis

An end-to-end Machine Learning pipeline that performs sentiment analysis on user reviews for popular Indonesian e-commerce apps. This project scrapes live data from the Google Play Store, preprocesses the text, and compares a classical Logistic Regression baseline against a fine-tuned DistilBERT transformer model.

 <!-- Replace with your own banner image -->

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Models Implemented](#models-implemented)
- [Performance Expectations](#performance-expectations)
- [File Structure](#file-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Pipeline](#how-to-run-the-pipeline)
- [API Usage Examples](#api-usage-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)


## Project Overview

The primary objective is to classify app reviews from Indonesian e-commerce giants **Tokopedia** and **Shopee** into three sentiment categories: **positive**, **negative**, or **neutral**. This project demonstrates a full MLOps lifecycle, from data acquisition and validation to model training, evaluation, and comparison, with a focus on creating production-ready, maintainable code.

## Architecture

The project follows a modular pipeline architecture, allowing each step to be executed independently.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │    │   Preprocessing  │    │   Model Training │
│                 │    │                  │    │                 │
│ Google Play     │───▶│ • Text Cleaning  │───▶│ • Baseline (LR) │
│ Store Reviews   │    │ • Tokenization   │    │ • Transformer   │
│ (Tokopedia,     │    │ • Sentiment      │    │   (DistilBERT)  │
│  Shopee)        │    │   Mapping        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐
│   Evaluation    │    │    Deployment    │
│                 │    │                  │
│ • Confusion     │◀───│ • Model Export   │
│   Matrix        │    │ • API Ready      │
│ • Classification│    │                  │
│   Report        │    │                  │
└─────────────────┘    └──────────────────┘
```


## Models Implemented

### 1. Baseline Model

- **Algorithm**: `LogisticRegression` with `TfidfVectorizer`
- **Features**: Bag-of-words with n-grams (1,2)
- **Optimization**: `GridSearchCV` for hyperparameter tuning
- **Purpose**: To establish a fast and computationally efficient performance benchmark.


### 2. Transformer Model

- **Architecture**: `distilbert-base-multilingual-cased`
- **Approach**: Fine-tuning a pre-trained model on the specific task.
- **Context**: Leverages deep contextual understanding for higher accuracy on nuanced text.
- **Purpose**: To compare against the baseline with a state-of-the-art approach.


## Performance Expectations

Based on typical results for this type of task, the expected performance is:


| Model | Accuracy | Weighted F1-Score | Avg. Training Time |
| :-- | :-- | :-- | :-- |
| **Logistic Regression** | 75–85% | 0.75–0.85 | < 5 minutes |
| **DistilBERT** | 80–90% | 0.80–0.90 | 15–30 minutes (GPU) |

## File Structure

```
sentiment_analysis_project/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/raw_reviews.csv
│   └── processed/processed_reviews.csv
├── scripts/
│   ├── data_scraper.py
│   ├── data_preprocessor.py
│   ├── baseline_model.py
│   ├── transformer_model.py
│   └── model_comparison.py
├── models/
│   ├── baseline_model.joblib
│   └── distilbert/
├── utils/
│   ├── logging_config.py
│   └── data_validation.py
├── results/
│   ├── figures/
│   ├── reports/
│   └── comparison_results.json
└── logs/
    └── pipeline_*.log
```


## Setup and Installation

Follow these steps carefully to create a stable environment and avoid common dependency issues.

### 1. Create and Activate a Virtual Environment

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```


### 2. Install PyTorch

To avoid common `DLL load failed` errors with TensorFlow on Windows, **install PyTorch first**.

```bash
# Install PyTorch (CPU version is sufficient)
pip install torch torchvision torchaudio
```


### 3. Install Project Requirements

```bash
pip install -r requirements.txt
```


### 4. Download NLTK Data

```bash
python -m nltk.downloader punkt punkt_tab stopwords
```


## How to Run the Pipeline

The `main.py` script orchestrates the entire process.

#### Run the Full Pipeline

```bash
python main.py
```


#### Run Specific Steps

Use the `--steps` argument to execute specific parts of the pipeline.
**Available steps**: `scrape`, `preprocess`, `baseline`, `transformer`, `compare`.

```bash
# Example: Run preprocessing and then train the baseline model
python main.py --steps preprocess baseline
```


#### Skip Data Scraping

If you already have data, use the `--skip-scraping` flag to resume from the next step.

```bash
# Example: Run all steps except scraping
python main.py --skip-scraping
```


## API Usage Examples

Once trained, the models can be easily loaded and used for inference.

```python
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Load Baseline Model ---
baseline_model = joblib.load('models/baseline_model.joblib')
prediction = baseline_model.predict(['aplikasi ini luar biasa!'])
print(f"Baseline Prediction: {prediction[^0]}")

# --- Load Transformer Model ---
tokenizer = AutoTokenizer.from_pretrained('models/distilbert')
model = AutoModelForSequenceClassification.from_pretrained('models/distilbert')

# Make prediction
inputs = tokenizer('aplikasi ini lemot dan sering error', return_tensors='pt')
with torch.no_grad():
    logits = model(**inputs).logits
    
predicted_class_id = logits.argmax().item()
# Assuming class mapping: {0: 'negative', 1: 'neutral', 2: 'positive'}
print(f"Transformer Prediction: {model.config.id2label[predicted_class_id]}")
```


## Troubleshooting

#### 1. `UnicodeEncodeError` on Windows

- **Error**: `'charmap' codec can't encode character...`
- **Cause**: The default Windows console cannot display emojis.
- **Solution**: The `utils/logging_config.py` in this repo is already Windows-compatible and removes emojis from log messages.


#### 2. `LookupError: Resource punkt_tab not found`

- **Error**: `LookupError: * Resource punkt_tab not found...`
- **Cause**: NLTK's tokenizer data is missing.
- **Solution**: Run the NLTK downloader command from **Step 4** of the installation guide.


#### 3. `ImportError: DLL load failed` (TensorFlow/Transformers)

- **Error**: `ImportError: DLL load failed while importing _pywrap_tensorflow_internal...`
- **Cause**: `transformers` defaulted to a broken TensorFlow backend.
- **Solution**: Install **PyTorch first** (see **Step 2** of setup). This forces `transformers` to use the PyTorch backend, bypassing the issue.

