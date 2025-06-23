import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = max(probabilities)
    return prediction, confidence

# Streamlit UI
st.title("Sentiment Analysis App")

# Real-time single text analysis
# Real-time + button-based single text analysis
st.write("### Analyze Single Text (Real-Time or Button)")

user_input = st.text_area("Enter text to analyze sentiment", height=150)
analyze_clicked = st.button("Analyze")

# Trigger analysis either when there's input or when the button is clicked
if user_input.strip() != "" or analyze_clicked:
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.3f}")



st.write("---")  # Separator line

# File upload section
st.write("### Or Upload a CSV File to Analyze Multiple Texts")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("CSV uploaded successfully!")
    
    if len(df.columns) == 1:
        text_col = df.columns[0]
    else:
        text_col = st.selectbox("Select the column containing the text", df.columns)
    
    df['Sentiment'] = df[text_col].apply(lambda x: predict_sentiment(str(x))[0])
    df['Confidence'] = df[text_col].apply(lambda x: predict_sentiment(str(x))[1])
    
    st.write("Sentiment analysis results:")
    st.dataframe(df[[text_col, 'Sentiment', 'Confidence']])
    
    # Convert DataFrame to CSV and enable download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download result.csv",
        data=csv,
        file_name='result.csv',
        mime='text/csv',
    )
