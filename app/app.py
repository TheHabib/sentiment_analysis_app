import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Pro",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .sentiment-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .positive-sentiment {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .negative-sentiment {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    
    .neutral-sentiment {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load('../models/sentiment_model.pkl')
        vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        st.stop()

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

def preprocess_text(text):
    """Enhanced text preprocessing with better handling"""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return ' '.join(tokens)
    except:
        return text

def predict_sentiment(text):
    """Predict sentiment with confidence score"""
    if not text or text.strip() == "":
        return "neutral", 0.0
    
    processed_text = preprocess_text(text)
    if not processed_text:
        return "neutral", 0.0
    
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = max(probabilities)
    return prediction, confidence

def get_sentiment_color(sentiment):
    """Get color based on sentiment"""
    colors = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#ffc107'
    }
    return colors.get(sentiment.lower(), '#6c757d')

def get_sentiment_emoji(sentiment):
    """Get emoji based on sentiment"""
    emojis = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emojis.get(sentiment.lower(), '🤔')

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎭 Sentiment Analysis Pro</h1>
    <p>Analyze emotions in text with AI-powered sentiment detection</p>
</div>
""", unsafe_allow_html=True)

# Load models
model, vectorizer = load_models()

# Sidebar with information
with st.sidebar:
    # GitHub link at the top
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <a href="https://github.com/TheHabib/sentiment_analysis_app" target="_blank" style="
            display: inline-block;
            background: linear-gradient(90deg, #24292e 0%, #586069 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
        ">
            🐙 View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 About This App")
    st.markdown("""
    This app uses machine learning to analyze the sentiment of text data:
    
    **Features:**
    - Real-time sentiment analysis
    - Batch processing via CSV upload
    - Confidence scoring
    - Interactive visualizations
    - Downloadable results
    
    **Sentiment Categories:**
    - 😊 Positive
    - 😞 Negative  
    - 😐 Neutral
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Usage Tips")
    st.markdown("""
    - Enter text in the input area for instant analysis
    - Upload CSV files for bulk analysis
    - Results include confidence scores
    - Download processed data as CSV
    """)

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Single Text Analysis")
    
    # Text input with enhanced styling
    user_input = st.text_area(
        "Enter your text here:",
        height=150,
        placeholder="Type or paste your text here to analyze its sentiment..."
    )
    
    # Analysis button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    st.markdown("### 📈 Quick Stats")
    
    # Real-time analysis
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        
        # Display results in styled card
        sentiment_class = f"{sentiment.lower()}-sentiment"
        emoji = get_sentiment_emoji(sentiment)
        
        st.markdown(f"""
        <div class="sentiment-card {sentiment_class}">
            <h3>{emoji} {sentiment.title()}</h3>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Text Length:</strong> {len(user_input)} characters</p>
            <p><strong>Word Count:</strong> {len(user_input.split())} words</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence meter
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': get_sentiment_color(sentiment)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Separator
st.markdown("---")

# File upload section
st.markdown("### 📁 Batch Analysis")

# Create upload area
upload_col1, upload_col2, upload_col3 = st.columns([1, 2, 1])

with upload_col2:
    st.markdown("""
    <div class="upload-section">
        <h4>📤 Upload CSV File</h4>
        <p>Upload a CSV file to analyze multiple texts at once</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file containing text data for batch sentiment analysis"
    )

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ CSV uploaded successfully! Found {len(df)} rows.")
        
        # Column selection
        if len(df.columns) == 1:
            text_col = df.columns[0]
            st.info(f"Using column: **{text_col}**")
        else:
            text_col = st.selectbox(
                "Select the column containing the text:",
                df.columns,
                help="Choose which column contains the text you want to analyze"
            )
        
        # Progress bar for analysis
        if st.button("🚀 Analyze All Texts", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze sentiments with progress tracking
            sentiments = []
            confidences = []
            
            for i, text in enumerate(df[text_col]):
                sentiment, confidence = predict_sentiment(str(text))
                sentiments.append(sentiment)
                confidences.append(confidence)
                
                # Update progress
                progress = (i + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing... {i + 1}/{len(df)} texts analyzed")
            
            # Add results to dataframe
            df['Sentiment'] = sentiments
            df['Confidence'] = confidences
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("### 📊 Analysis Results")
            
            # Summary statistics
            sentiment_counts = df['Sentiment'].value_counts()
            avg_confidence = df['Confidence'].mean()
            
            # Create metrics row
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("Total Texts", len(df))
            
            with metric_cols[1]:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with metric_cols[2]:
                positive_count = sentiment_counts.get('positive', 0)
                st.metric("Positive", positive_count)
            
            with metric_cols[3]:
                negative_count = sentiment_counts.get('negative', 0)
                st.metric("Negative", negative_count)
            
            # Visualization
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Sentiment distribution pie chart
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'positive': '#28a745',
                        'negative': '#dc3545',
                        'neutral': '#ffc107'
                    }
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with viz_col2:
                # Confidence distribution histogram
                fig_hist = px.histogram(
                    df,
                    x='Confidence',
                    title='Confidence Score Distribution',
                    nbins=20,
                    color_discrete_sequence=['#667eea']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Results table
            st.markdown("### 📋 Detailed Results")
            
            # Add color coding to the dataframe display
            def highlight_sentiment(row):
                colors = {
                    'positive': 'background-color: #d4edda',
                    'negative': 'background-color: #f8d7da',
                    'neutral': 'background-color: #fff3cd'
                }
                return [colors.get(row['Sentiment'], '')] * len(row)
            
            styled_df = df[[text_col, 'Sentiment', 'Confidence']].style.apply(highlight_sentiment, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download section
            st.markdown("### 💾 Download Results")
            
            # Prepare download data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_data = df.to_csv(index=False).encode('utf-8')
            
            download_cols = st.columns([1, 2, 1])
            with download_cols[1]:
                st.download_button(
                    label="📥 Download Complete Results",
                    data=csv_data,
                    file_name=f'sentiment_analysis_results_{timestamp}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted with text data.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #6c757d;">
    <p>Built with ❤️ using Streamlit • Enhanced UI for better user experience</p>
</div>
""", unsafe_allow_html=True)