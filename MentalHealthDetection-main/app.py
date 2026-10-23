
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

model = DistilBertForSequenceClassification.from_pretrained('./results')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
labels = ['anger', 'fear', 'joy', 'sadness'] # Adjusted labels based on the training data

st.title("🧠 Mental Health Detection from Text")
st.write("Detect emotional tone (e.g., sadness, anxiety) from user text.")

user_input = st.text_area("Enter a social media post:")

if st.button("Analyze"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs).item()
    st.write(f"**Detected Emotion:** {labels[pred]}")
