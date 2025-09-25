# config.py
import os
from pathlib import Path

class Config:
    """Configuration settings for the sentiment analysis project"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Data collection settings
    APPS_TO_SCRAPE = {
        'tokopedia': 'com.tokopedia.tkpd',
        'shopee': 'com.shopee.id'
    }
    REVIEWS_PER_APP = 1000
    SCRAPING_LANGUAGE = 'id'  # Indonesian
    SCRAPING_COUNTRY = 'id'   # Indonesia
    
    # Preprocessing settings
    MIN_TOKEN_LENGTH = 3
    MIN_REVIEW_TOKENS = 3
    MAX_FEATURES_TFIDF = 10000
    
    # Model settings
    BASELINE_MODEL = {
        'random_state': 42,
        'max_iter': 1000,
        'class_weight': 'balanced'
    }
    
    TRANSFORMER_MODEL = {
        'model_name': 'distilbert-base-multilingual-cased',
        'max_length': 128,
        'batch_size': 16,
        'num_epochs': 3,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'weight_decay': 0.01
    }
    
    # Evaluation settings
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Sentiment mapping
    SENTIMENT_MAPPING = {
        1: 'negative',
        2: 'negative',
        3: 'neutral',
        4: 'positive',
        5: 'positive'
    }
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def get_model_path(cls, model_name):
        """Get the full path for a model"""
        return cls.MODELS_DIR / f"{model_name}.joblib"
    
    @classmethod
    def get_data_path(cls, filename, processed=True):
        """Get the full path for a data file"""
        if processed:
            return cls.PROCESSED_DATA_DIR / filename
        else:
            return cls.RAW_DATA_DIR / filename