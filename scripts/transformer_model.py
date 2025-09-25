# scripts/transformer_model.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerClassifier:
    def __init__(self, model_name='distilbert-base-multilingual-cased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
        # Label mapping
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def prepare_dataset(self, texts, labels):
        """Prepare dataset for training"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=128
            )
        
        # Convert labels to numeric
        numeric_labels = [self.label2id[label] for label in labels]
        
        # Create HuggingFace dataset
        dataset = HFDataset.from_dict({
            'text': texts.tolist(),
            'label': numeric_labels
        })
        
        # Tokenize
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, X_train, y_train, X_val, y_val, output_dir='models/distilbert'):
        """Fine-tune the DistilBERT model"""
        logger.info("Preparing datasets...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(X_train, y_train)
        val_dataset = self.prepare_dataset(X_val, y_val)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy='steps',
            eval_steps=500,
            save_strategy='steps',
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            report_to=None,  # Disable wandb logging
            seed=42
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        logger.info("Starting fine-tuning...")
        self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def predict(self, texts):
        """Make predictions on new texts"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
        
        # Convert to labels
        predicted_labels = [self.id2label[label.item()] for label in predicted_labels]
        
        return predicted_labels
    
    def evaluate(self, X_test, y_test, save_path=None):
        """Evaluate the model"""
        predictions = self.predict(X_test)
        
        # Classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_test, predictions, output_dict=True)
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=['negative', 'neutral', 'positive'])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['negative', 'neutral', 'positive'],
                   yticklabels=['negative', 'neutral', 'positive'])
        plt.title('Confusion Matrix - Transformer Model (DistilBERT)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(f'{save_path}/transformer_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report, cm

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
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train transformer model
    transformer = TransformerClassifier()
    transformer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    report, cm = transformer.evaluate(X_test, y_test, save_path='results')