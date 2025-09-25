# utils/logging_config.py - Windows Compatible Version
import logging
import sys
from pathlib import Path
from datetime import datetime
import locale

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration for the project with Windows compatibility
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    
    # Create formatter without emojis for Windows compatibility
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with UTF-8 encoding for Windows
    try:
        # Try to set UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # For Windows, ensure proper encoding
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        
        root_logger.addHandler(console_handler)
    except Exception:
        # Fallback to basic handler if UTF-8 setup fails
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception:
            # Fallback without explicit encoding
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    # Suppress some verbose libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return root_logger

class ModelLogger:
    """Custom logger for model training and evaluation"""
    
    def __init__(self, name, log_file=None):
        self.logger = logging.getLogger(name)
        if log_file:
            # Create logs directory
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            try:
                handler = logging.FileHandler(log_file, encoding='utf-8')
            except Exception:
                handler = logging.FileHandler(log_file)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_data_info(self, df, stage=""):
        """Log dataset information"""
        self.logger.info(f"Dataset Info {stage}:")
        self.logger.info(f"  Shape: {df.shape}")
        self.logger.info(f"  Columns: {list(df.columns)}")
        
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            self.logger.info(f"  Sentiment Distribution:")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"    {sentiment}: {count} ({percentage:.2f}%)")
    
    def log_model_performance(self, model_name, metrics):
        """Log model performance metrics"""
        self.logger.info(f"Performance Results for {model_name}:")
        self.logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        
        if 'weighted avg' in metrics:
            weighted = metrics['weighted avg']
            self.logger.info(f"  Weighted Avg:")
            self.logger.info(f"    Precision: {weighted.get('precision', 'N/A'):.4f}")
            self.logger.info(f"    Recall: {weighted.get('recall', 'N/A'):.4f}")
            self.logger.info(f"    F1-Score: {weighted.get('f1-score', 'N/A'):.4f}")
    
    def log_training_progress(self, epoch, loss, metrics=None):
        """Log training progress"""
        self.logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")
        if metrics:
            for metric_name, value in metrics.items():
                self.logger.info(f"  {metric_name}: {value:.4f}")

# Usage example
if __name__ == "__main__":
    # Setup main logging
    logger = setup_logging(
        log_level=logging.INFO,
        log_file='logs/sentiment_analysis.log'
    )
    
    # Test logging
    logger.info("Logging system initialized successfully")
    
    # Test model logger
    model_logger = ModelLogger("test_model", "logs/model_training.log")
    model_logger.logger.info("Model logger test successful")