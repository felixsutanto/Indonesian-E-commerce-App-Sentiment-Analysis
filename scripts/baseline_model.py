# scripts/baseline_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineClassifier:
    def __init__(self):
        self.pipeline = None
        self.best_params = None
        
    def create_pipeline(self):
        """Create ML pipeline with TF-IDF and Logistic Regression"""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words=None  # We already handled stopwords
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
        return pipeline
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Use GridSearchCV to find optimal hyperparameters"""
        logger.info("Starting hyperparameter optimization...")
        
        pipeline = self.create_pipeline()
        
        # Define parameter grid
        param_grid = {
            'tfidf__max_features': [5000, 10000, 15000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.pipeline = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return self.pipeline
    
    def train(self, X_train, y_train):
        """Train the model with optimal hyperparameters"""
        if self.pipeline is None:
            self.optimize_hyperparameters(X_train, y_train)
        else:
            self.pipeline.fit(X_train, y_train)
        
        return self.pipeline
    
    def evaluate(self, X_test, y_test, save_path=None):
        """Evaluate the model and generate reports"""
        y_pred = self.pipeline.predict(X_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['negative', 'neutral', 'positive'],
                   yticklabels=['negative', 'neutral', 'positive'])
        plt.title('Confusion Matrix - Baseline Model (Logistic Regression)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(f'{save_path}/baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report, cm
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/processed_reviews.csv')
    
    # Prepare data
    X = df['cleaned_content']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train baseline model
    baseline = BaselineClassifier()
    baseline.train(X_train, y_train)
    
    # Evaluate
    report, cm = baseline.evaluate(X_test, y_test, save_path='results')
    
    # Save model
    baseline.save_model('models/baseline_model.joblib')