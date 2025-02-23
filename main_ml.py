import pandas as pd
import numpy as np
import re
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

class LawPredictor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Store cases for exact matching
        self.cases = []
        
        # TF-IDF for similarity fallback
        self.tfidf = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            analyzer='word',
            token_pattern=r'\w+',
            min_df=1
        )

    def normalize_text(self, text):
        """Normalize text for exact matching"""
        text = str(text).lower().strip()
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        # Convert numbers to words for better matching
        text = re.sub(r'(\d+)\s*km/?h', r'\1 kilometers per hour', text)
        text = re.sub(r'(\d+)', r'\1', text)  # Keep numbers as is
        return text

    def preprocess_text(self, text):
        """Preprocess text for similarity matching"""
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def extract_key_info(self, text):
        """Extract key information from text"""
        text = text.lower()
        
        # Extract speeds with units
        speeds = re.findall(r'(\d+)\s*km/?h', text)
        speeds.extend(re.findall(r'(\d+)\s*kilometers?\s+per\s+hours?', text))
        speeds = [int(s) for s in speeds]
        
        # Extract amounts with currency
        amounts = re.findall(r'(?:rs\.?)?\s*(\d+(?:,\d+)?)', text)
        amounts = [int(a.replace(',', '')) for a in amounts]
        
        # Extract key phrases
        key_info = {
            'max_speed': max(speeds) if speeds else 0,
            'max_amount': max(amounts) if amounts else 0,
            'has_accident': bool(re.search(r'accident|crash|collide|collision', text)),
            'has_injury': bool(re.search(r'injur|hurt|wound|fatal', text)),
            'has_alcohol': bool(re.search(r'drunk|alcohol|intoxicat', text)),
            'has_license': bool(re.search(r'licen[cs]e|permit', text)),
            'has_parking': bool(re.search(r'park|standing|halt', text)),
            'location_type': None
        }
        
        # Determine location type
        if re.search(r'school|education|student', text):
            key_info['location_type'] = 'school_zone'
        elif re.search(r'hospital|medical|clinic', text):
            key_info['location_type'] = 'hospital_zone'
        elif re.search(r'residential|housing|colony', text):
            key_info['location_type'] = 'residential'
        elif re.search(r'highway|expressway|freeway', text):
            key_info['location_type'] = 'highway'
        
        return key_info

    def train(self, data_path):
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding='latin1')
        
        print("\nInitial Data Statistics:")
        print(f"Total cases: {len(df)}")
        
        # Store all cases with normalized text
        self.cases = []
        for _, row in df.iterrows():
            case = {
                'original_text': row['Case Facts'],
                'normalized_text': self.normalize_text(row['Case Facts']),
                'preprocessed_text': self.preprocess_text(row['Case Facts']),
                'key_info': self.extract_key_info(row['Case Facts']),
                'law': str(row['Relevant Law']).strip(),
                'outcome': str(row['Outcome']).strip()
            }
            self.cases.append(case)
        
        # Prepare TF-IDF for similarity fallback
        texts = [case['preprocessed_text'] for case in self.cases]
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        
        print("\nTraining completed. Ready for predictions.")

    def predict(self, case_facts):
        # Normalize input text
        normalized_input = self.normalize_text(case_facts)
        input_key_info = self.extract_key_info(case_facts)
        
        best_match = None
        best_score = 0
        
        for case in self.cases:
            score = 0
            
            # Exact text match
            if case['normalized_text'] == normalized_input:
                return {
                    'law': case['law'],
                    'outcome': case['outcome'],
                    'match_type': 'exact'
                }
            
            # Key information match
            if case['key_info']['max_speed'] == input_key_info['max_speed'] and input_key_info['max_speed'] > 0:
                score += 3
            if case['key_info']['max_amount'] == input_key_info['max_amount'] and input_key_info['max_amount'] > 0:
                score += 2
            
            # Context matches
            if case['key_info']['has_accident'] == input_key_info['has_accident']:
                score += 1
            if case['key_info']['has_injury'] == input_key_info['has_injury']:
                score += 1
            if case['key_info']['has_alcohol'] == input_key_info['has_alcohol']:
                score += 1
            if case['key_info']['has_license'] == input_key_info['has_license']:
                score += 1
            if case['key_info']['has_parking'] == input_key_info['has_parking']:
                score += 1
            if case['key_info']['location_type'] == input_key_info['location_type'] and input_key_info['location_type']:
                score += 2
            
            # Text similarity score
            text_similarity = self.compute_similarity(case['preprocessed_text'], self.preprocess_text(case_facts))
            score += text_similarity * 3
            
            if score > best_score:
                best_score = score
                best_match = case
        
        if best_score >= 8:  # High confidence threshold
            return {
                'law': best_match['law'],
                'outcome': best_match['outcome'],
                'match_type': 'high_confidence'
            }
        elif best_score >= 5:  # Medium confidence
            return {
                'law': best_match['law'],
                'outcome': best_match['outcome'],
                'match_type': 'medium_confidence'
            }
        else:  # Low confidence, use most similar case
            input_vector = self.tfidf.transform([self.preprocess_text(case_facts)])
            similarities = (self.tfidf_matrix * input_vector.T).toarray().flatten()
            most_similar_idx = similarities.argmax()
            
            return {
                'law': self.cases[most_similar_idx]['law'],
                'outcome': self.cases[most_similar_idx]['outcome'],
                'match_type': 'fallback'
            }
    
    def compute_similarity(self, text1, text2):
        """Compute text similarity score between 0 and 1"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0

    def standardize_law(self, law_text):
        if pd.isna(law_text):
            return "Unknown Section"
        return str(law_text).strip()

    def standardize_outcome(self, outcome_text):
        if pd.isna(outcome_text):
            return "Unknown"
        return str(outcome_text).strip()

    def save_model(self):
        joblib.dump({
            'cases': self.cases,
            'tfidf': self.tfidf,
            'tfidf_matrix': self.tfidf_matrix
        }, 'model.joblib')

    @classmethod
    def load_model(cls):
        predictor = cls()
        models = joblib.load('model.joblib')
        predictor.cases = models['cases']
        predictor.tfidf = models['tfidf']
        predictor.tfidf_matrix = models['tfidf_matrix']
        return predictor

if __name__ == '__main__':
    predictor = LawPredictor()
    predictor.train('draft_mini_dataset.csv')
