# utils/data_validation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """Comprehensive data validation and quality assessment"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_raw_data(self, df: pd.DataFrame) -> Dict:
        """Validate raw scraped data"""
        results = {
            'total_records': len(df),
            'missing_values': {},
            'data_types': {},
            'unique_values': {},
            'anomalies': []
        }
        
        # Check required columns
        required_columns = ['content', 'score', 'app_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results['anomalies'].append(f"Missing required columns: {missing_columns}")
        
        # Missing values analysis
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            results['missing_values'][column] = {
                'count': missing_count,
                'percentage': round(missing_percentage, 2)
            }
        
        # Data types
        results['data_types'] = df.dtypes.to_dict()
        
        # Unique values for categorical columns
        categorical_cols = ['app_name', 'score']
        for col in categorical_cols:
            if col in df.columns:
                results['unique_values'][col] = df[col].value_counts().to_dict()
        
        # Score validation
        if 'score' in df.columns:
            invalid_scores = df[~df['score'].between(1, 5)]['score'].value_counts()
            if not invalid_scores.empty:
                results['anomalies'].append(f"Invalid scores found: {invalid_scores.to_dict()}")
        
        # Text length analysis
        if 'content' in df.columns:
            df['content_length'] = df['content'].astype(str).str.len()
            results['text_statistics'] = {
                'min_length': df['content_length'].min(),
                'max_length': df['content_length'].max(),
                'mean_length': round(df['content_length'].mean(), 2),
                'median_length': df['content_length'].median()
            }
            
            # Check for very short or very long reviews
            very_short = len(df[df['content_length'] < 10])
            very_long = len(df[df['content_length'] > 1000])
            results['text_quality'] = {
                'very_short_reviews': very_short,
                'very_long_reviews': very_long
            }
        
        self.validation_results['raw_data'] = results
        return results
    
    def validate_processed_data(self, df: pd.DataFrame) -> Dict:
        """Validate processed data"""
        results = {
            'total_records': len(df),
            'sentiment_distribution': {},
            'text_quality': {},
            'anomalies': []
        }
        
        # Check required columns for processed data
        required_columns = ['cleaned_content', 'sentiment', 'token_count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results['anomalies'].append(f"Missing processed columns: {missing_columns}")
        
        # Sentiment distribution
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            total = len(df)
            
            for sentiment, count in sentiment_counts.items():
                results['sentiment_distribution'][sentiment] = {
                    'count': count,
                    'percentage': round((count / total) * 100, 2)
                }
            
            # Check for class imbalance
            min_class = sentiment_counts.min()
            max_class = sentiment_counts.max()
            imbalance_ratio = max_class / min_class
            
            if imbalance_ratio > 3:
                results['anomalies'].append(f"Class imbalance detected: {imbalance_ratio:.2f}:1")
        
        # Text quality after preprocessing
        if 'cleaned_content' in df.columns:
            # Empty content after cleaning
            empty_content = df['cleaned_content'].str.strip().eq('').sum()
            if empty_content > 0:
                results['anomalies'].append(f"Empty content after cleaning: {empty_content} records")
            
            # Token count analysis
            if 'token_count' in df.columns:
                results['text_quality'] = {
                    'min_tokens': df['token_count'].min(),
                    'max_tokens': df['token_count'].max(),
                    'mean_tokens': round(df['token_count'].mean(), 2),
                    'median_tokens': df['token_count'].median()
                }
                
                # Very short reviews after preprocessing
                very_short_processed = len(df[df['token_count'] < 3])
                if very_short_processed > 0:
                    results['anomalies'].append(f"Very short reviews after processing: {very_short_processed}")
        
        self.validation_results['processed_data'] = results
        return results
    
    def create_data_quality_report(self, df_raw: pd.DataFrame = None, 
                                  df_processed: pd.DataFrame = None,
                                  save_path: str = None):
        """Generate comprehensive data quality report"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Data Quality Assessment Report', fontsize=16, fontweight='bold')
        
        if df_raw is not None:
            # Raw data score distribution
            if 'score' in df_raw.columns:
                df_raw['score'].value_counts().sort_index().plot(kind='bar', ax=axes[0,0])
                axes[0,0].set_title('Raw Data: Score Distribution')
                axes[0,0].set_xlabel('Rating Score')
                axes[0,0].set_ylabel('Count')
            
            # Raw data text length distribution
            if 'content' in df_raw.columns:
                text_lengths = df_raw['content'].astype(str).str.len()
                axes[0,1].hist(text_lengths, bins=50, alpha=0.7)
                axes[0,1].set_title('Raw Data: Text Length Distribution')
                axes[0,1].set_xlabel('Character Count')
                axes[0,1].set_ylabel('Frequency')
            
            # Missing values heatmap
            missing_data = df_raw.isnull().sum()
            if missing_data.sum() > 0:
                missing_data.plot(kind='bar', ax=axes[0,2])
                axes[0,2].set_title('Raw Data: Missing Values')
                axes[0,2].set_xlabel('Columns')
                axes[0,2].set_ylabel('Missing Count')
                axes[0,2].tick_params(axis='x', rotation=45)
        
        if df_processed is not None:
            # Processed data sentiment distribution
            if 'sentiment' in df_processed.columns:
                df_processed['sentiment'].value_counts().plot(kind='bar', ax=axes[1,0])
                axes[1,0].set_title('Processed Data: Sentiment Distribution')
                axes[1,0].set_xlabel('Sentiment')
                axes[1,0].set_ylabel('Count')
            
            # Token count distribution
            if 'token_count' in df_processed.columns:
                axes[1,1].hist(df_processed['token_count'], bins=30, alpha=0.7)
                axes[1,1].set_title('Processed Data: Token Count Distribution')
                axes[1,1].set_xlabel('Token Count')
                axes[1,1].set_ylabel('Frequency')
            
            # App distribution
            if 'app_name' in df_processed.columns:
                df_processed['app_name'].value_counts().plot(kind='bar', ax=axes[1,2])
                axes[1,2].set_title('Processed Data: App Distribution')
                axes[1,2].set_xlabel('App Name')
                axes[1,2].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/data_quality_report.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_validation_summary(self):
        """Print a summary of all validation results"""
        
        print("\n" + "="*80)
        print("DATA VALIDATION SUMMARY")
        print("="*80)
        
        for data_type, results in self.validation_results.items():
            print(f"\n{data_type.upper()} VALIDATION:")
            print(f"  Total Records: {results['total_records']:,}")
            
            # Missing values
            if 'missing_values' in results:
                print("  Missing Values:")
                for col, missing_info in results['missing_values'].items():
                    if missing_info['count'] > 0:
                        print(f"    {col}: {missing_info['count']} ({missing_info['percentage']}%)")
            
            # Sentiment distribution
            if 'sentiment_distribution' in results:
                print("  Sentiment Distribution:")
                for sentiment, info in results['sentiment_distribution'].items():
                    print(f"    {sentiment.title()}: {info['count']} ({info['percentage']}%)")
            
            # Text statistics
            if 'text_statistics' in results:
                print("  Text Length Statistics:")
                stats = results['text_statistics']
                print(f"    Min: {stats['min_length']} chars")
                print(f"    Max: {stats['max_length']} chars") 
                print(f"    Mean: {stats['mean_length']} chars")
                print(f"    Median: {stats['median_length']} chars")
            
            # Anomalies
            if results['anomalies']:
                print("  ⚠️  Anomalies Detected:")
                for anomaly in results['anomalies']:
                    print(f"    - {anomaly}")
        
        print("\n" + "="*80)

# Usage example
if __name__ == "__main__":
    # Example usage
    validator = DataValidator()
    
    # Simulate some data for testing
    raw_data = pd.DataFrame({
        'content': ['Great app!', 'Terrible experience', '', 'Good but has bugs'],
        'score': [5, 1, 3, 4],
        'app_name': ['tokopedia', 'shopee', 'tokopedia', 'shopee']
    })
    
    processed_data = pd.DataFrame({
        'cleaned_content': ['great app', 'terrible experience', 'good has bugs'],
        'sentiment': ['positive', 'negative', 'positive'],
        'token_count': [2, 2, 3],
        'app_name': ['tokopedia', 'shopee', 'shopee']
    })
    
    # Run validations
    raw_results = validator.validate_raw_data(raw_data)
    processed_results = validator.validate_processed_data(processed_data)
    
    # Print summary
    validator.print_validation_summary()