import streamlit as st
import requests
from newspaper import Article
import re
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

nltk.download("punkt")

# ==============================
# Load DL Model
# ==============================
@st.cache_resource
def load_dl_model():
    model_name = "mrm8488/bert-tiny-finetuned-fake-news-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return clf

dl_model = load_dl_model()

# ==============================
# API Config
# ==============================
API_KEY = "AIzaSyDlnSBUgoN2m94xmaFY2WIT-GjYC8MOUUg"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def query_api(text: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": f"Decide if this news is real or fake. Reply only with 'REAL' or 'FAKE'. News: {text}"}]}]
    }
    try:
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"Error from API: {e}"

# ==============================
# URL Article Extractor
# ==============================
def extract_text_from_url(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        return f"Error extracting article: {e}"

# ==============================
# Clean Article Text
# ==============================
def clean_article_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    seen = set()
    for line in lines:
        line = line.strip()
        if len(line) > 30 and line not in seen:  # remove tiny junk/duplicates
            cleaned.append(line)
            seen.add(line)
    return " ".join(cleaned)

# ==============================
# Final Prediction
# ==============================
def get_final_prediction(text: str) -> str:
    # DL Model Prediction
    dl_result = dl_model(text[:512])[0]  # only first 512 tokens
    dl_label = dl_result["label"]
    dl_score = dl_result["score"]

    # API Prediction
    api_result = query_api(text)

    # Final Decision: use API result directly
    if "FAKE" in api_result.upper():
        return "🔴 FAKE NEWS"
    elif "REAL" in api_result.upper():
        return "🟢 REAL NEWS"
    else:
        return f"🤖 Unclear: {api_result}"

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection")

input_type = st.radio("Choose Input Type", ["Text", "URL"])

if input_type == "Text":
    text_input = st.text_area("Enter news article text here")
    if st.button("Run Fake News Detection"):
        if text_input.strip():
            verdict = get_final_prediction(clean_article_text(text_input))
            st.subheader("Final Verdict:")
            st.write(verdict)
        else:
            st.warning("Please enter some text.")

elif input_type == "URL":
    url_input = st.text_input("Enter news article URL")
    if st.button("Run Fake News Detection"):
        if url_input.strip():
            raw_text = extract_text_from_url(url_input)
            article_text = clean_article_text(raw_text)
            st.text_area("📄 Extracted Article", article_text, height=300)
            verdict = get_final_prediction(article_text)
            st.subheader("Final Verdict:")
            st.write(verdict)
        else:
            st.warning("Please enter a URL.")
