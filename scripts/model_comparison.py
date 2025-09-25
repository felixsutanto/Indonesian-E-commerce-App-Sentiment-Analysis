# scripts/model_comparison.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_compare_models():
    """Load both models and compare their performance"""
    
    # Load data
    df = pd.read_csv('data/processed/processed_reviews.csv')
    X = df['cleaned_content']
    y = df['sentiment']
    
    # Split data (same split as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    
    # Baseline Model Evaluation
    logger.info("Evaluating Baseline Model...")
    baseline_model = joblib.load('models/baseline_model.joblib')
    baseline_pred = baseline_model.predict(X_test)
    baseline_report = classification_report(y_test, baseline_pred, output_dict=True)
    results['baseline'] = baseline_report
    
    # Transformer Model Evaluation
    logger.info("Evaluating Transformer Model...")
    from transformer_model import TransformerClassifier
    
    transformer = TransformerClassifier()
    # Load the fine-tuned model
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    transformer.model = AutoModelForSequenceClassification.from_pretrained('models/distilbert')
    transformer.tokenizer = AutoTokenizer.from_pretrained('models/distilbert')
    
    transformer_pred = transformer.predict(X_test)
    transformer_report = classification_report(y_test, transformer_pred, output_dict=True)
    results['transformer'] = transformer_report
    
    # Create comparison visualization
    create_comparison_plots(results)
    
    # Print comparison summary
    print_comparison_summary(results)
    
    return results

def create_comparison_plots(results):
    """Create comparison plots for both models"""
    
    # Extract metrics for plotting
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['negative', 'neutral', 'positive']
    
    # Prepare data for plotting
    comparison_data = []
    for model_name, report in results.items():
        for class_name in classes:
            for metric in metrics:
                comparison_data.append({
                    'model': model_name,
                    'class': class_name,
                    'metric': metric,
                    'value': report[class_name][metric.replace('-', '_')]
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        metric_data = comparison_df[comparison_df['metric'] == metric]
        
        # Pivot for easier plotting
        pivot_data = metric_data.pivot(index='class', columns='model', values='value')
        
        # Create grouped bar plot
        pivot_data.plot(kind='bar', ax=axes[i], width=0.8)
        axes[i].set_title(f'{metric.title()} Comparison')
        axes[i].set_xlabel('Sentiment Class')
        axes[i].set_ylabel(metric.title())
        axes[i].legend(title='Model')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Overall performance comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    overall_metrics = []
    for model_name, report in results.items():
        overall_metrics.append({
            'model': model_name.title(),
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        })
    
    overall_df = pd.DataFrame(overall_metrics)
    overall_df.set_index('model').plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Overall Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.legend(title='Metrics')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig('results/overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_comparison_summary(results):
    """Print a summary comparison of both models"""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    for model_name, report in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        print(f"  Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"  Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        print("  Per-class Performance:")
        for class_name in ['negative', 'neutral', 'positive']:
            class_metrics = report[class_name]
            print(f"    {class_name.title()}:")
            print(f"      Precision: {class_metrics['precision']:.4f}")
            print(f"      Recall: {class_metrics['recall']:.4f}")
            print(f"      F1-Score: {class_metrics['f1-score']:.4f}")
    
    # Determine better model
    baseline_f1 = results['baseline']['weighted avg']['f1-score']
    transformer_f1 = results['transformer']['weighted avg']['f1-score']
    
    print(f"\n{'='*80}")
    if transformer_f1 > baseline_f1:
        improvement = ((transformer_f1 - baseline_f1) / baseline_f1) * 100
        print(f"WINNER: Transformer Model performs {improvement:.2f}% better than Baseline")
    else:
        decline = ((baseline_f1 - transformer_f1) / baseline_f1) * 100
        print(f"WINNER: Baseline Model performs {decline:.2f}% better than Transformer")
    print("="*80)

if __name__ == "__main__":
    results = load_and_compare_models()
    
    # Save results to JSON
    with open('results/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)