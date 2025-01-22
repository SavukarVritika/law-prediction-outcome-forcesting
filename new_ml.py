import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import joblib
import re

class LawPredictor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.law_model = KNeighborsClassifier(
            n_neighbors=1,
            metric='cosine'
        )
        self.outcome_model = KNeighborsClassifier(
            n_neighbors=1,
            metric='cosine'
        )
        self.laws = None
        self.outcomes = None

    def preprocess_text(self, text):
        text = str(text).lower()
        # Replace specific patterns
        text = re.sub(r'(\d+)\s*km/?h', r'speed_\1_kmph', text)
        text = re.sub(r'section\s+(\d+[A-Za-z]*)', r'section_\1', text)
        text = re.sub(r'article\s+(\d+[A-Za-z]*)', r'article_\1', text)
        # Remove special characters but keep the patterns we created
        text = re.sub(r'[^a-z0-9_\s]', ' ', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def train(self, data_path):
        # Load data
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding='latin1')
        
        # Store original labels
        self.laws = df['Relevant Law'].values
        self.outcomes = df['Outcome'].values
        
        # Preprocess text and convert to TF-IDF
        X = self.tfidf.fit_transform(df['Case Facts'].apply(self.preprocess_text))
        
        # Train models using indices as labels
        y_law = np.arange(len(df))
        y_outcome = np.arange(len(df))
        
        self.law_model.fit(X, y_law)
        self.outcome_model.fit(X, y_outcome)
        
        # Save models and data
        joblib.dump({
            'law_model': self.law_model,
            'outcome_model': self.outcome_model,
            'tfidf': self.tfidf,
            'laws': self.laws,
            'outcomes': self.outcomes
        }, 'model.joblib')
        
        return "Model trained successfully!"
    
    def predict(self, case_facts):
        # Preprocess input text
        text_processed = self.preprocess_text(case_facts)
        X = self.tfidf.transform([text_processed])
        
        # Get nearest neighbor indices
        law_idx = self.law_model.predict(X)[0]
        outcome_idx = self.outcome_model.predict(X)[0]
        
        # Get original labels
        predicted_law = self.laws[law_idx]
        predicted_outcome = self.outcomes[outcome_idx]
        
        return {
            'law': predicted_law,
            'outcome': predicted_outcome
        }

    @classmethod
    def load_model(cls):
        predictor = cls()
        models = joblib.load('model.joblib')
        predictor.law_model = models['law_model']
        predictor.outcome_model = models['outcome_model']
        predictor.tfidf = models['tfidf']
        predictor.laws = models['laws']
        predictor.outcomes = models['outcomes']
        return predictor

if __name__ == "__main__":
    predictor = LawPredictor()
    
    # Load and preprocess data
    try:
        df = pd.read_csv("draft_mini_dataset.csv", encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv("draft_mini_dataset.csv", encoding='latin1')
    
    # Prepare features and indices
    X = predictor.tfidf.fit_transform(df['Case Facts'].apply(predictor.preprocess_text))
    y = np.arange(len(df))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate
    predictor.laws = df['Relevant Law'].values
    predictor.outcomes = df['Outcome'].values
    
    predictor.law_model.fit(X_train, y_train)
    predictor.outcome_model.fit(X_train, y_train)
    
    # Calculate accuracy using exact matches
    law_preds = predictor.laws[predictor.law_model.predict(X_test)]
    outcome_preds = predictor.outcomes[predictor.outcome_model.predict(X_test)]
    
    law_accuracy = np.mean(law_preds == df['Relevant Law'].values[y_test])
    outcome_accuracy = np.mean(outcome_preds == df['Outcome'].values[y_test])
    
    print(f"Law Prediction Accuracy: {law_accuracy:.2%}")
    print(f"Outcome Prediction Accuracy: {outcome_accuracy:.2%}")
    
    # Train final model on full dataset
    predictor.train("draft_mini_dataset.csv")
