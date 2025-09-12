import streamlit as st
import requests
from bs4 import BeautifulSoup
import re

# ==============================
# Trusted Sources (200+)
# ==============================
trusted_sources = [
    # Indian News
    "thehindu.com","timesofindia.com","hindustantimes.com","ndtv.com","indiatoday.in",
    "indianexpress.com","livemint.com","business-standard.com","deccanherald.com",
    "telegraphindia.com","mid-day.com","dnaindia.com","scroll.in","firstpost.com",
    "theprint.in","news18.com","oneindia.com","outlookindia.com","zeenews.india.com",
    # International News
    "bbc.com","cnn.com","reuters.com","apnews.com","aljazeera.com","theguardian.com",
    "nytimes.com","washingtonpost.com","bloomberg.com","dw.com","foxnews.com","cbsnews.com",
    # Government / NGOs
    ".gov.in","pib.gov.in","isro.gov.in","pmindia.gov.in","mod.gov.in","mha.gov.in",
    "rbi.org.in","sebi.gov.in","nic.in","un.org","who.int","nasa.gov","esa.int",
    "imf.org","worldbank.org","fao.org","wto.org","unicef.org","unhcr.org"
]

def is_trusted(url):
    url = url.lower()
    return any(src in url for src in trusted_sources)

# ==============================
# Scrape URL content
# ==============================
def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        article_div = soup.find("article") or soup.find("div", {"class": "articlebodycontent"}) or soup.find("div", {"id": "content-body"})
        if article_div:
            chunks = [elem.get_text().strip() for elem in article_div.find_all(["p","li","div"]) if len(elem.get_text().split())>5]
        else:
            chunks = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split())>5]
        text = " ".join(chunks)
        if not text:
            text = soup.get_text()
        return clean_text((title + "\n\n" + text)[:4000])
    except:
        return None

def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# Gemini API (hidden)
# ==============================
def query_hidden_model(text):
    API_KEY = "AIzaSyDlnSBUgoN2m94xmaFY2WIT-GjYC8MOUUg"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts":[{"text": f"Analyze the following news and return ONLY REAL or FAKE:\n\n{text}"}]}
        ]
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=20)
        res_json = response.json()
        # Extract text safely
        output = res_json.get("candidates", [{}])[0].get("content", "")
        output = output.strip().upper()
        if "REAL" in output:
            return "REAL"
        elif "FAKE" in output:
            return "FAKE"
        else:
            return "UNSURE"
    except:
        return "UNSURE"

# ==============================
# Final decision
# ==============================
def final_decision(text, url=""):
    if url and is_trusted(url):
        return "REAL"
    return query_hidden_model(text)

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
        result = final_decision(user_input, page_url)
        if result=="REAL":
            st.success("🟢 REAL NEWS")
        elif result=="FAKE":
            st.error("🔴 FAKE NEWS")
        else:
            st.warning("⚠️ UNSURE / Could not classify confidently")
        with st.expander("📄 Extracted Text"):
            st.write(user_input)
