# main.py - Enhanced version with comprehensive error handling
import os
import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from config import Config
from utils.logging_config import setup_logging, ModelLogger
from utils.data_validation import DataValidator

def create_directory_structure():
    """Create the complete project directory structure"""
    try:
        Config.create_directories()
        
        # Additional directories
        additional_dirs = [
            'notebooks',
            'utils',
            'logs',
            'results/figures',
            'results/reports'
        ]
        
        for directory in additional_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        logger.error(f"Failed to create directory structure: {str(e)}")
        return False

def run_data_scraping():
    """Execute data scraping with error handling"""
    logger.info("=" * 50)
    logger.info("STEP 1: DATA SCRAPING")
    logger.info("=" * 50)
    
    try:
        from scripts.data_scraper import AppReviewScraper
        
        scraper = AppReviewScraper()
        reviews_df = scraper.scrape_all_apps(reviews_per_app=Config.REVIEWS_PER_APP)
        
        if reviews_df.empty:
            logger.error("No reviews were scraped")
            return False
        
        # Save raw data
        output_path = Config.get_data_path('raw_reviews.csv', processed=False)
        reviews_df.to_csv(output_path, index=False)
        
        logger.info(f" Successfully scraped {len(reviews_df)} reviews")
        logger.info(f"   Saved to: {output_path}")
        
        # Log app distribution
        app_counts = reviews_df['app_name'].value_counts()
        for app, count in app_counts.items():
            logger.info(f"   {app}: {count} reviews")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error in data scraping: {str(e)}")
        logger.error("Please install required packages: pip install google-play-scraper")
        return False
    except Exception as e:
        logger.error(f"Error in data scraping: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def run_data_preprocessing():
    """Execute data preprocessing with validation"""
    logger.info("=" * 50)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("=" * 50)
    
    try:
        import pandas as pd
        from scripts.data_preprocessor import ReviewPreprocessor
        
        # Load raw data
        raw_data_path = Config.get_data_path('raw_reviews.csv', processed=False)
        if not raw_data_path.exists():
            logger.error(f"Raw data file not found: {raw_data_path}")
            return False
        
        raw_df = pd.read_csv(raw_data_path)
        logger.info(f"Loaded {len(raw_df)} raw reviews")
        
        # Validate raw data
        validator = DataValidator()
        validator.validate_raw_data(raw_df)
        
        # Preprocess data
        preprocessor = ReviewPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(raw_df)
        
        if processed_df.empty:
            logger.error("Preprocessing resulted in empty dataset")
            return False
        
        # Validate processed data
        validator.validate_processed_data(processed_df)
        
        # Save processed data
        output_path = Config.get_data_path('processed_reviews.csv', processed=True)
        processed_df.to_csv(output_path, index=False)
        
        logger.info(f" Successfully processed {len(processed_df)} reviews")
        logger.info(f"   Saved to: {output_path}")
        
        # Log data loss
        data_loss = len(raw_df) - len(processed_df)
        data_loss_pct = (data_loss / len(raw_df)) * 100
        logger.info(f"   Data loss: {data_loss} records ({data_loss_pct:.2f}%)")
        
        # Generate data quality report
        validator.create_data_quality_report(
            df_raw=raw_df,
            df_processed=processed_df,
            save_path='results'
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def run_baseline_training():
    """Execute baseline model training"""
    logger.info("=" * 50)
    logger.info("STEP 3: BASELINE MODEL TRAINING")
    logger.info("=" * 50)
    
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from scripts.baseline_model import BaselineClassifier
        
        # Load processed data
        data_path = Config.get_data_path('processed_reviews.csv', processed=True)
        df = pd.read_csv(data_path)
        
        # Prepare data
        X = df['cleaned_content']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train baseline model
        baseline = BaselineClassifier()
        baseline.train(X_train, y_train)
        
        # Evaluate
        report, cm = baseline.evaluate(X_test, y_test, save_path='results')
        
        # Save model
        model_path = Config.get_model_path('baseline_model')
        baseline.save_model(model_path)
        
        logger.info(" Baseline model training completed")
        logger.info(f"   Model saved to: {model_path}")
        logger.info(f"   Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in baseline model training: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def run_transformer_training():
    """Execute transformer model training"""
    logger.info("=" * 50)
    logger.info("STEP 4: TRANSFORMER MODEL TRAINING")
    logger.info("=" * 50)
    
    try:
        import pandas as pd
        import torch
        from sklearn.model_selection import train_test_split
        from scripts.transformer_model import TransformerClassifier
        
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load processed data
        data_path = Config.get_data_path('processed_reviews.csv', processed=True)
        df = pd.read_csv(data_path)
        
        # Prepare data
        X = df['cleaned_content']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y
        )
        
        # Further split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=Config.VALIDATION_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y_train
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train transformer model
        transformer = TransformerClassifier()
        transformer.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        report, cm = transformer.evaluate(X_test, y_test, save_path='results')
        
        logger.info(" Transformer model training completed")
        logger.info(f"   Model saved to: models/distilbert/")
        logger.info(f"   Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error - missing transformers library: {str(e)}")
        logger.error("Please install: pip install transformers torch datasets")
        return False
    except Exception as e:
        logger.error(f"Error in transformer model training: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def run_model_comparison():
    """Execute model comparison and generate reports"""
    logger.info("=" * 50)
    logger.info("STEP 5: MODEL COMPARISON")
    logger.info("=" * 50)
    
    try:
        from scripts.model_comparison import load_and_compare_models
        
        # Run comparison
        results = load_and_compare_models()
        
        logger.info(" Model comparison completed")
        logger.info("   Results saved to: results/comparison_results.json")
        logger.info("   Figures saved to: results/")
        
        # Log final comparison
        baseline_f1 = results['baseline']['weighted avg']['f1-score']
        transformer_f1 = results['transformer']['weighted avg']['f1-score']
        
        if transformer_f1 > baseline_f1:
            improvement = ((transformer_f1 - baseline_f1) / baseline_f1) * 100
            logger.info(f" Transformer model outperforms baseline by {improvement:.2f}%")
        else:
            decline = ((baseline_f1 - transformer_f1) / baseline_f1) * 100
            logger.info(f" Baseline model performs {decline:.2f}% better than transformer")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def generate_final_report():
    """Generate comprehensive final report"""
    logger.info("=" * 50)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 50)
    
    try:
        import json
        import pandas as pd
        from datetime import datetime
        
        # Load results
        results_path = Path('results/comparison_results.json')
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            logger.warning("Comparison results not found")
            return False
        
        # Load processed data for statistics
        data_path = Config.get_data_path('processed_reviews.csv', processed=True)
        df = pd.read_csv(data_path)
        
        # Generate report
        report_content = f"""
# Sentiment Analysis Project - Final Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview

This project implemented an end-to-end sentiment analysis pipeline for Indonesian e-commerce app reviews.

## Dataset Summary

- **Total Reviews:** {len(df):,}
- **Apps Analyzed:** {', '.join(df['app_name'].unique())}
- **Sentiment Distribution:**
"""
        
        for sentiment, count in df['sentiment'].value_counts().items():
            pct = (count / len(df)) * 100
            report_content += f"  - {sentiment.title()}: {count:,} ({pct:.2f}%)\n"
        
        report_content += f"""

## Model Performance Summary

### Baseline Model (Logistic Regression + TF-IDF)
- **Accuracy:** {results['baseline']['accuracy']:.4f}
- **Weighted F1-Score:** {results['baseline']['weighted avg']['f1-score']:.4f}
- **Macro F1-Score:** {results['baseline']['macro avg']['f1-score']:.4f}

### Transformer Model (DistilBERT)
- **Accuracy:** {results['transformer']['accuracy']:.4f}
- **Weighted F1-Score:** {results['transformer']['weighted avg']['f1-score']:.4f}
- **Macro F1-Score:** {results['transformer']['macro avg']['f1-score']:.4f}

## Key Findings

"""
        
        baseline_f1 = results['baseline']['weighted avg']['f1-score']
        transformer_f1 = results['transformer']['weighted avg']['f1-score']
        
        if transformer_f1 > baseline_f1:
            improvement = ((transformer_f1 - baseline_f1) / baseline_f1) * 100
            report_content += f"- The transformer model outperformed the baseline by {improvement:.2f}%\n"
        else:
            decline = ((baseline_f1 - transformer_f1) / baseline_f1) * 100
            report_content += f"- The baseline model performed {decline:.2f}% better than the transformer\n"
        
        report_content += f"""
- Average review length: {df['token_count'].mean():.1f} tokens
- Most challenging class to predict: {min(results['baseline'].keys()[:3], key=lambda x: results['baseline'][x]['f1-score'])}

## Files Generated

- `data/processed/processed_reviews.csv` - Cleaned dataset
- `models/baseline_model.joblib` - Trained baseline model
- `models/distilbert/` - Fine-tuned transformer model
- `results/model_comparison.png` - Performance comparison charts
- `results/data_quality_report.png` - Data quality assessment

## Recommendations for Production

1. **Model Selection:** {'Use transformer model for higher accuracy' if transformer_f1 > baseline_f1 else 'Use baseline model for efficiency'}
2. **Data Quality:** Continue monitoring review quality and filtering
3. **Model Updates:** Retrain periodically with new data
4. **Performance Monitoring:** Track model performance on new reviews

---
*Report generated automatically by the sentiment analysis pipeline*
"""
        
        # Save report
        report_path = Path('results/reports/final_report.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f" Final report generated: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating final report: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('--steps', nargs='+', 
                       choices=['scrape', 'preprocess', 'baseline', 'transformer', 'compare', 'all'],
                       default=['all'],
                       help='Steps to execute')
    parser.add_argument('--skip-scraping', action='store_true',
                       help='Skip data scraping (use existing data)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    global logger
    logger = setup_logging(
        log_level=getattr(logging, args.log_level),
        log_file=log_file
    )
    
    logger.info(" Starting Sentiment Analysis Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create directory structure
    if not create_directory_structure():
        logger.error("Failed to create directory structure")
        return 1
    
    # Determine steps to execute
    if 'all' in args.steps:
        steps = ['scrape', 'preprocess', 'baseline', 'transformer', 'compare']
    else:
        steps = args.steps
    
    if args.skip_scraping and 'scrape' in steps:
        steps.remove('scrape')
        logger.info("Skipping data scraping as requested")
    
    # Execute pipeline steps
    success = True
    step_functions = {
        'scrape': run_data_scraping,
        'preprocess': run_data_preprocessing,
        'baseline': run_baseline_training,
        'transformer': run_transformer_training,
        'compare': run_model_comparison
    }
    
    for step in steps:
        if step in step_functions:
            if not step_functions[step]():
                logger.error(f"Pipeline failed at step: {step}")
                success = False
                break
        else:
            logger.warning(f"Unknown step: {step}")
    
    if success:
        # Generate final report
        generate_final_report()
        
        logger.info(" Pipeline completed successfully!")
        logger.info(f"Check the results in the 'results/' directory")
        logger.info(f"Full log available at: {log_file}")
        return 0
    else:
        logger.error(" Pipeline failed!")
        return 1

if __name__ == "__main__":
    import logging
    exit_code = main()
    sys.exit(exit_code)