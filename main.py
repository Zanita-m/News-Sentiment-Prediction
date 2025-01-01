import streamlit as st

# to start streamlit, use > streamlit run main.py
# to stop streamlit, use > Get-Process streamlit ; > Stop-Process streamlit

import re
import pickle
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load XGBoost model and label encoder from pickle
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Load additional stopwords
def load_stopwords():
    with open("rare_words-12.txt", "r") as file:
        rare_words = file.read().splitlines()
    with open("stopwords-wc.txt", "r") as file:
        wc_stopwords = file.read().splitlines()
    return set(rare_words + wc_stopwords)

stopwords_indonesia = set(stopwords.words("indonesian"))
stopwords_additional = load_stopwords()

# Initialize Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Initialize Vader Sentiment Analyzer for English text
analyzer = SentimentIntensityAnalyzer()

# Preprocessing Functions
def preprocess_text(text):
    # 1. Implementing Regex: Clean text
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove non-alphanumeric characters

    # 2. Implementing Lemmatization / Stemming
    text = stemmer.stem(text)  # Apply stemming

    # 3. Implementing Stopwords Removal
    text = " ".join([word for word in text.split() if word not in stopwords_indonesia and word not in stopwords_additional])

    return text

# Sentiment Prediction Function
def predict_sentiment(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Apply TF-IDF transformation
    transformed_text = vectorizer.transform([processed_text])
    
    # Prediction
    prediction = model.predict(transformed_text)

    # Decode label
    sentiment = label_encoder.inverse_transform(prediction)
    
    return sentiment[0]

# ------- Streamlit Frontend ----------
st.title("News Data Sentiment Analyzer")
st.write("**Khusus untuk data mengenai berita Pemilu Pilpres 2024**")

article_text = st.text_area("Masukkan teks artikel disini:")

if st.button("Analyze Sentiment"):
    if article_text.strip():
        st.write("1. Implementing Regex...")
        st.write("2. Implementing Lemmatization...")
        st.write("3. Implementing Stopwords...")
        
        # Sentiment Analysis
        sentiment = predict_sentiment(article_text)
        
        # Display result
        st.success(f"The sentiment is: {sentiment}")
    else:
        st.error("Mohon masukkan teks!")
