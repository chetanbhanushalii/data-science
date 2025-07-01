import streamlit as st
import pickle
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 🔧 Custom Preprocessing Class (must match train_model.py)
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = re.sub(r"[^a-zA-Z]", " ", text.lower()) # Lowercase and remove non-alphabetic characters
        tokens = nltk.word_tokenize(text) # Tokenize
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and word not in string.punctuation]
        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text)

# ✅ Load pipeline (now TextPreprocessor is defined)
pipeline = pickle.load(open("nb_pipeline_with_cleaning.pkl", "rb"))

# 🎯 Streamlit UI
st.set_page_config(page_title="Review Classifier", page_icon="🧠")
st.title("🧠 Course Review Sentiment Classifier")
st.write("Enter a course review, and we'll classify it as **Good** or **Bad**!")

review = st.text_area("Write your course review here:")

if st.button("Classify"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        prediction = pipeline.predict(pd.Series([review]))[0]
        prob = pipeline.predict_proba(pd.Series([review])).max()

        if prediction == 1:
            st.success(f"🟢 Good Review ({prob:.2%} confidence)")
        elif prediction == 0:
            st.error(f"🔴 Bad Review!! ({prob:.2%} confidence)")
        else:
            st.error("❌ Unable to classify the review. Please try again.")
