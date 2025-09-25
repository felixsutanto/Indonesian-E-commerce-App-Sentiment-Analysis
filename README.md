# E-commerce Review Sentiment Analysis

This project provides an end-to-end Machine Learning pipeline to perform sentiment analysis on user reviews for popular Indonesian e-commerce applications. It scrapes live data from the Google Play Store, preprocesses the text, and compares two models: a classical Logistic Regression baseline and a fine-tuned DistilBERT transformer.

 <!-- It's good practice to add a banner image -->

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Methodology](#methodology)
- [File Structure](#file-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Pipeline](#how-to-run-the-pipeline)
- [Troubleshooting](#troubleshooting)


## Project Overview

The primary goal of this project is to classify app reviews from Indonesian e-commerce giants like **Tokopedia** and **Shopee** into three sentiment categories: **positive**, **negative**, or **neutral**. This demonstrates a full MLOps lifecycle, from data acquisition and validation to model training, evaluation, and comparison.

## Features

- **Automated Data Scraping**: Fetches the most recent reviews using the `google-play-scraper` library.
- **Robust Preprocessing Pipeline**: Cleans text data by handling missing values, removing special characters, and applying tokenization with stopword removal for Indonesian.
- **Dual-Model Comparison**:

1. **Baseline Model**: A `LogisticRegression` classifier with `TfidfVectorizer` to establish a performance benchmark.
2. **Transformer Model**: A fine-tuned `distilbert-base-multilingual-cased` model to leverage contextual understanding for higher accuracy.
- **Comprehensive Evaluation**: Generates classification reports, confusion matrices, and a final comparison report for both models.
- **Modular \& Configurable**: The entire pipeline is controlled via a central `main.py` script and a `config.py` file, allowing for easy modification and step-by-step execution.


## Methodology

1. **Data Acquisition**: Scrapes 1,000 recent Indonesian reviews each for Tokopedia and Shopee.
2. **Preprocessing**: Reviews are cleaned, and a sentiment label (`positive`, `negative`, `neutral`) is created based on the star rating (1-2 stars: negative, 3: neutral, 4-5: positive).
3. **Baseline Model**: A TF-IDF vectorizer converts text into numerical features, which are then used to train a Logistic Regression model.
4. **Transformer Model**: The pre-trained DistilBERT model is fine-tuned on the review dataset, leveraging its deep understanding of language nuances.
5. **Evaluation**: Both models are evaluated on a held-out test set, and their performance (precision, recall, F1-score) is compared to determine the most effective approach.

## File Structure

```
sentiment_analysis_project/
├── main.py                          # Main execution script
├── config.py                        # Configuration settings
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── data/
│   ├── raw/raw_reviews.csv          # Original scraped data
│   └── processed/processed_reviews.csv # Cleaned and labeled data
├── scripts/
│   ├── data_scraper.py              # Google Play Store scraper
│   ├── data_preprocessor.py         # Text preprocessing pipeline
│   ├── baseline_model.py            # Logistic Regression classifier
│   ├── transformer_model.py         # DistilBERT fine-tuning
│   └── model_comparison.py          # Model evaluation & comparison
├── models/
│   ├── baseline_model.joblib        # Saved baseline model
│   └── distilbert/                  # Fine-tuned transformer
├── utils/
│   ├── logging_config.py            # Logging utilities
│   └── data_validation.py           # Data quality checks
├── results/
│   ├── figures/                     # Generated plots & charts
│   ├── reports/                     # Analysis reports
│   └── comparison_results.json      # Model performance metrics
└── logs/
    └── pipeline_*.log               # Execution logs
```


## Setup and Installation

Follow these steps carefully to create a stable environment and avoid common dependency issues, especially on Windows.

### 1. Create and Activate a Virtual Environment

Using a virtual environment is crucial for managing dependencies and preventing conflicts.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```


### 2. Install Core ML Libraries (PyTorch)

The `transformers` library can cause issues with TensorFlow on Windows (`DLL load failed`). The most reliable fix is to install **PyTorch first**, which `transformers` will then use as its default backend.

```bash
# Install PyTorch (CPU version is sufficient)
pip install torch torchvision torchaudio
```


### 3. Install Project Requirements

With PyTorch installed, you can now safely install the remaining packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```


### 4. Download NLTK Data

The preprocessing script requires data packages from the Natural Language Toolkit (NLTK).

```bash
python -m nltk.downloader punkt punkt_tab stopwords
```


## How to Run the Pipeline

The `main.py` script orchestrates the entire pipeline.

### Running the Full Pipeline

To execute all steps from data scraping to final comparison:

```bash
python main.py
```


### Running Specific Steps

Use the `--steps` argument to run specific parts of the pipeline. This is useful for debugging or re-running a failed step.

**Available steps**: `scrape`, `preprocess`, `baseline`, `transformer`, `compare`.

```bash
# Example: Run preprocessing and train the baseline model
python main.py --steps preprocess baseline
```


### Skipping Data Scraping

If you have already scraped data, use the `--skip-scraping` flag to avoid re-downloading it.

```bash
# Example: Run the entire pipeline except for the scraping step
python main.py --skip-scraping
```


## Troubleshooting

Here are solutions to common errors encountered during setup.

### 1. `UnicodeEncodeError` on Windows

- **Error**: `UnicodeEncodeError: 'charmap' codec can't encode character...`
- **Cause**: The default Windows console cannot display Unicode characters like emojis.
- **Solution**: The `utils/logging_config.py` in this repository is already configured to be Windows-compatible by removing emojis and ensuring UTF-8 encoding.


### 2. `LookupError: Resource punkt_tab not found`

- **Error**: `LookupError: * Resource punkt_tab not found...`
- **Cause**: NLTK's `punkt_tab` tokenizer data is missing.
- **Solution**: This is resolved by running the NLTK downloader command in **Step 4** of the installation guide.


### 3. `ImportError: DLL load failed` (TensorFlow/Transformers)

- **Error**: `ImportError: DLL load failed while importing _pywrap_tensorflow_internal...`
- **Cause**: The `transformers` library defaulted to a broken TensorFlow backend. This is a common issue on Windows.
- **Solution**: **Install PyTorch first**, as described in **Step 2** of the installation guide. This forces `transformers` to use PyTorch, bypassing the TensorFlow issue entirely.

