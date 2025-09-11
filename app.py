import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
import re

# ==============================
# Set your OpenAI API Key here
# ==============================
openai.api_key = "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"  # Replace with your key

# ==============================
# Web Scraping Function
# ==============================
def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        # Get title and paragraphs
        title = soup.title.string if soup.title else ""
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text()) > 20]
        text = title + "\n\n" + "\n".join(paragraphs)
        return text
    except:
        return None

# ==============================
# ChatGPT Prediction Function
# ==============================
def chatgpt_predict(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant."},
                {"role": "user", "content": f"Classify the following news as REAL or FAKE, and explain briefly why:\n\n{text}"}
            ],
            temperature=0
        )
        verdict = response.choices[0].message.content
        return verdict
    except Exception as e:
        return f"Error: {e}"

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="📰 ChatGPT Fake News Detector", layout="wide")
st.title("📰 ChatGPT Fake News Detection App")

input_type = st.radio("Choose Input Type", ["Text", "URL"])
user_input = ""

if input_type == "Text":
    user_input = st.text_area("Enter news text here", height=200)
elif input_type == "URL":
    page_url = st.text_input("Enter news article URL")
    if page_url:
        scraped_text = scrape_url(page_url)
        if scraped_text:
            st.text_area("Extracted Article", scraped_text, height=300)
            user_input = scraped_text
        else:
            st.warning("⚠️ Could not scrape the URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter valid text or URL.")
    else:
        with st.spinner("Analyzing with ChatGPT..."):
            prediction = chatgpt_predict(user_input)
            st.subheader("Prediction & Explanation:")
            st.write(prediction)
