import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_model()

# Set title
st.title("Sentiment Classifier (BERT - Hugging Face)")
st.write("Enter a review and get its predicted sentiment (positive or negative).")

# Text input
user_input = st.text_area("Review Text", "I loved the movie!")

# Predict function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probs).item()
    return "Positive 😀" if prediction == 1 else "Negative 😠", probs[0][prediction].item()

# Button
if st.button("Predict"):
    label, confidence = predict_sentiment(user_input)
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence:** {confidence:.2f}")
