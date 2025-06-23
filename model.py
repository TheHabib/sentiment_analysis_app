# Complete Sentiment Analysis Implementation with Dataset
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt_tab')

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)



# Text preprocessing function
def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Main execution
def main():
    print("Creating sentiment analysis dataset...")
    
    # Create dataset
    # Load dataset from CSV
    df = pd.read_csv('sentiment_dataset.csv')
    print(f"Dataset loaded with {len(df)} samples")

    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Display sample data
    print("\nSample data:")
    print(df.head(10))
    
    # Preprocess text
    print("\nPreprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Feature extraction
    print("Extracting features...")
    X = df['processed_text']
    y = df['sentiment']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train multiple models
    models = {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    model_results = {}
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Save the best model and components
    print("Saving model components...")
    joblib.dump(best_model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    
    # Create a simple prediction function for testing
    def predict_sentiment(text):
        processed_text = preprocess_text(text)
        text_tfidf = vectorizer.transform([processed_text])
        prediction = best_model.predict(text_tfidf)[0]
        probabilities = best_model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities)
        return prediction, confidence
    
    # Test the model with new examples
    print("\nTesting the model with new examples:")
    test_cases = [
        "This product is absolutely amazing! I love it!",
        "Terrible quality. I hate this product.",
        "It's okay, nothing special but does the job.",
        "Best purchase ever! Highly recommend!",
        "Worst experience of my life. Avoid at all costs!",
        "The product is decent for its price range."
    ]
    
    for text in test_cases:
        sentiment, confidence = predict_sentiment(text)
        print(f"Text: '{text}'")
        print(f"Predicted: {sentiment} (confidence: {confidence:.3f})")
        print("-" * 50)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Sentiment distribution
    plt.subplot(2, 2, 1)
    sentiment_counts = df['sentiment'].value_counts()
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('Sentiment Distribution in Dataset')
    
    # Plot 2: Model comparison
    plt.subplot(2, 2, 2)
    model_names = list(model_results.keys())
    accuracies = [model_results[name]['accuracy'] for name in model_names]
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Plot 3: Text length distribution
    plt.subplot(2, 2, 3)
    text_lengths = df['text'].str.len()
    plt.hist(text_lengths, bins=20, alpha=0.7)
    plt.title('Text Length Distribution')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    
    # Plot 4: Confusion matrix for best model
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(y_test, model_results[best_model_name]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nModel training complete!")
    print(f"Files saved:")
    print("- sentiment_dataset.csv (dataset)")
    print("- sentiment_model.pkl (trained model)")
    print("- tfidf_vectorizer.pkl (feature vectorizer)")
    print("- sentiment_analysis_results.png (visualization)")
    
    return df, best_model, vectorizer, predict_sentiment

if __name__ == "__main__":
    df, model, vectorizer, predict_func = main()