import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import json

# ==============================
# API Key (hidden in env if possible)
# ==============================
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDlnSBUgoN2m94xmaFY2WIT-GjYC8MOUUg")

# ==============================
# Load DL Models (cover models)
# ==============================
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
    tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_roberta_model():
    return pipeline("zero-shot-classification", model="roberta-large-mnli")

bert_pipeline = load_bert_model()
roberta_pipeline = load_roberta_model()

# ==============================
# Text Cleaning
# ==============================
def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# Web Scraping
# ==============================
def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        article_div = soup.find("article") or soup.find("div", {"class": "articlebodycontent"}) or soup.find("div", {"id": "content-body"})
        if article_div:
            chunks = [elem.get_text().strip() for elem in article_div.find_all(["p","li","div"]) if len(elem.get_text().split()) > 5]
        else:
            chunks = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split()) > 5]
        text = " ".join(chunks)
        if not text:
            text = soup.get_text()
        return clean_text((title + "\n\n" + text)[:4000])
    except:
        return None

# ==============================
# Trusted Sources
# ==============================
trusted_sources = [
    "thehindu.com","timesofindia.com","hindustantimes.com","ndtv.com","indiatoday.in",
    "indianexpress.com","livemint.com","business-standard.com","deccanherald.com",
    "telegraphindia.com","mid-day.com","dnaindia.com","scroll.in","firstpost.com",
    "theprint.in","news18.com","oneindia.com","outlookindia.com","zeenews.india.com",
    "cnnnews18.com","economictimes.indiatimes.com","financialexpress.com","siasat.com",
    "newindianexpress.com","tribuneindia.com","asianage.com","bharattimes.com",
    "freepressjournal.in","morningindia.in","abplive.com","newsable.asianetnews.com",
    "bbc.com","cnn.com","reuters.com","apnews.com","aljazeera.com","theguardian.com",
    "nytimes.com","washingtonpost.com","bloomberg.com","dw.com","foxnews.com","cbsnews.com",
    "nbcnews.com","abcnews.go.com","sky.com","france24.com","rt.com","sputniknews.com",
    "npr.org","telegraph.co.uk","thetimes.co.uk","independent.co.uk","globaltimes.cn",
    "china.org.cn","cbc.ca","abc.net.au","smh.com.au","japantimes.co.jp","lemonde.fr",
    "elpais.com","spiegel.de","tagesschau.de","asiatimes.com","straitstimes.com",
    "thejakartapost.com","theglobeandmail.com","latimes.com","chicagotribune.com",
    "startribune.com","nydailynews.com","financialtimes.com","forbes.com","thehill.com",
    "vox.com","buzzfeednews.com","huffpost.com","usatoday.com","euronews.com","cnbc.com",
    "barrons.com","time.com","economist.com","foreignpolicy.com","dailytelegraph.com.au",
    "smh.com.au","thesun.co.uk","dailymail.co.uk",".gov.in","pib.gov.in","isro.gov.in",
    "pmindia.gov.in","mha.gov.in","rbi.org.in","sebi.gov.in","nic.in","mohfw.gov.in",
    "nasa.gov","esa.int","imf.org","worldbank.org","un.org","who.int","unicef.org",
    "unhcr.org","redcross.org","cdc.gov","nih.gov","usa.gov","gov.uk","canada.ca",
    "australia.gov.au","japan.go.jp","europa.eu","ec.europa.eu","wfp.org","oecd.org"
]

def is_trusted(url):
    url = url.lower()
    return any(src in url for src in trusted_sources)

# ==============================
# API Final Prediction
# ==============================
def api_final_decision(text):
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Decide if the following news is REAL or FAKE. Answer only 'REAL' or 'FAKE'.\n\n{text}"}
                    ]
                }
            ]
        }
        r = requests.post(f"{url}?key={API_KEY}", headers=headers, data=json.dumps(payload))
        if r.status_code == 200:
            response = r.json()
            out_text = response["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
            if "REAL" in out_text:
                return "REAL"
            elif "FAKE" in out_text:
                return "FAKE"
        return "UNSURE"
    except Exception as e:
        return f"ERROR: {e}"

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection")

input_type = st.radio("Choose Input Type", ["Text", "URL"])

user_input = ""
page_url = ""

if input_type == "Text":
    user_input = st.text_area("Enter news text here", height=200)
elif input_type == "URL":
    page_url = st.text_input("Enter news article URL")
    if page_url:
        scraped = scrape_url(page_url)
        if scraped:
            st.text_area("Extracted Article", scraped, height=300)
            user_input = scraped
        else:
            st.warning("⚠️ Could not scrape the URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter valid text or URL.")
    else:
        try:
            # Final decision comes from API
            verdict = api_final_decision(user_input)

            st.subheader("Final Verdict:")
            if verdict == "REAL":
                st.success("🟢 REAL NEWS")
            elif verdict == "FAKE":
                st.error("🔴 FAKE NEWS")
            else:
                st.warning("⚠️ UNSURE")

            with st.expander("📄 Extracted Text"):
                st.write(user_input)

        except Exception as e:
            st.error(f"⚠️ Error during analysis: {e}")
