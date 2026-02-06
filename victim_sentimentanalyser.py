import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import joblib

class FakeReviewVictimModel:
    def __init__(self):
        self.sentiment_pipe = None
        self.fake_pipe = None
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return "missing review"
        return str(text).lower().strip()
    
    def simple_sentiment_baseline(self, text):
        pos_words = ['great', 'love', 'amazing', 'perfect', 'excellent', 'best']
        neg_words = ['bad', 'terrible', 'worst', 'hate', 'awful', 'trash']
        score = sum(word in text for word in pos_words) - sum(word in text for word in neg_words)
        return 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
    
    def load_and_prepare_data(self, csv_path="fake_reviews_dataset.csv"):
        """Fixed for actual dataset columns: category, rating, label, text_"""
        df = pd.read_csv(csv_path)
        print(f"Dataset: {df.shape[0]} reviews")
        print("Columns:", df.columns.tolist())
        print("Label distribution:\n", df['label'].value_counts())
        
        # Use columns: text_ and label (CG/OR)
        df['clean_text'] = df['text_'].apply(self.preprocess_text)  
        df['sentiment'] = df['clean_text'].apply(self.simple_sentiment_baseline)
        
        # Map CG=0 (Real), OR=1 (Fake/Generated) 
        df['is_fake'] = (df['label'] == 'OR').astype(int)  
        
        print("Fake/Real distribution:\n", df['is_fake'].value_counts())
        return df
    
    def train(self, csv_path="fake_reviews_dataset.csv"):
        df = self.load_and_prepare_data(csv_path)
        
        # Split data
        X_train, X_test, y_sent_train, y_sent_test = train_test_split(
            df['clean_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
        )
        _, _, y_fake_train, y_fake_test = train_test_split(
            df['clean_text'], df['is_fake'], test_size=0.2, random_state=42
        )
        
        print("\nTraining Sentiment Model...")
        self.sentiment_pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        self.sentiment_pipe.fit(X_train, y_sent_train)
        sent_f1 = f1_score(y_sent_test, self.sentiment_pipe.predict(X_test), average='weighted')
        print(f"Sentiment F1: {sent_f1:.3f}")
        
        print("\nTraining Fake Detection Model...")
        self.fake_pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
            ('clf', LogisticRegression(random_state=42, max_iter=1000, C=0.1))
        ])
        self.fake_pipe.fit(X_train, y_fake_train)
        fake_f1 = f1_score(y_fake_test, self.fake_pipe.predict(X_test))
        print(f"Fake F1: {fake_f1:.3f}")
        
        # Save models
        joblib.dump(self.sentiment_pipe, "sentiment_victim.pkl")
        joblib.dump(self.fake_pipe, "fake_victim.pkl")
        print("\n Models saved!")
        
        return sent_f1, fake_f1
    
    def predict(self, review_text: str) -> dict:
        clean_text = self.preprocess_text(review_text)
        
        sent_pred = self.sentiment_pipe.predict([clean_text])[0]
        sent_probs = self.sentiment_pipe.predict_proba([clean_text])
        sent_conf = np.max(sent_probs)
        
        fake_pred = self.fake_pipe.predict([clean_text])[0]
        fake_prob = self.fake_pipe.predict_proba([clean_text])[0][1]
        
        return {
            "review_text": review_text,
            "sentiment": sent_pred,
            "sentiment_confidence": float(sent_conf),
            "is_fake": bool(fake_pred),
            "fake_probability": float(fake_prob)  # OR probability (generated/fake)
        }

# RUN
if __name__ == "__main__":
    victim = FakeReviewVictimModel()
    sent_f1, fake_f1 = victim.train("fake_reviews_dataset.csv")
    
    # Test
    result = victim.predict("This product is AMAZING!!!!")
    print("\n Test:")
    print(result)
