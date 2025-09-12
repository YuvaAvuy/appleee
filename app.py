import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import json

# ==============================
# Gemini API Key & Endpoint
# ==============================
GEMINI_API_KEY = "AIzaSyDlnSBUgoN2m94xmaFY2WIT-GjYC8MOUUg"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# ==============================
# Trusted Sources (200+)
# ==============================
trusted_sources = [
    # Indian News
    "thehindu.com","timesofindia.com","hindustantimes.com","ndtv.com","indiatoday.in",
    "indianexpress.com","livemint.com","business-standard.com","deccanherald.com",
    "telegraphindia.com","mid-day.com","dnaindia.com","scroll.in","firstpost.com",
    "theprint.in","news18.com","oneindia.com","outlookindia.com","zeenews.india.com",
    "cnnnews18.com","economictimes.indiatimes.com","financialexpress.com","siasat.com",
    "newindianexpress.com","tribuneindia.com","asianage.com","bharattimes.com",
    "freepressjournal.in","morningindia.in","abplive.com","newsable.asianetnews.com",
    # International News
    "bbc.com","cnn.com","reuters.com","apnews.com","aljazeera.com","theguardian.com",
    "nytimes.com","washingtonpost.com","bloomberg.com","dw.com","foxnews.com","cbsnews.com",
    "nbcnews.com","abcnews.go.com","sky.com","france24.com","rt.com","sputniknews.com",
    "npr.org","telegraph.co.uk","thetimes.co.uk","independent.co.uk","globaltimes.cn",
    "china.org.cn","cbc.ca","abc.net.au","smh.com.au","japantimes.co.jp","lemonde.fr",
    "elpais.com","derstandard.at","spiegel.de","tagesschau.de","asiatimes.com",
    "straitstimes.com","thaiworldview.com","thejakartapost.com","thestandard.com.hk",
    "sbs.com.au","hawaiinewsnow.com","theglobeandmail.com","irishnews.com","latimes.com",
    "chicagotribune.com","startribune.com","nydailynews.com","financialtimes.com",
    "forbes.com","thehill.com","vox.com","buzzfeednews.com","huffpost.com","usatoday.com",
    "teleSURenglish.net","euronews.com","al-monitor.com","news.com.au","cnbc.com",
    "barrons.com","time.com","foreignpolicy.com","economist.com","foreignaffairs.com",
    "dailytelegraph.com.au","smh.com.au","thesun.co.uk","dailymail.co.uk",
    # Indian Government
    ".gov.in","pib.gov.in","isro.gov.in","pmindia.gov.in","mod.gov.in","mha.gov.in",
    "rbi.org.in","sebi.gov.in","nic.in","mohfw.gov.in","moef.gov.in","meity.gov.in",
    "railway.gov.in","dgca.gov.in","drdo.gov.in","indianrailways.gov.in","education.gov.in",
    "scienceandtech.gov.in","urbanindia.nic.in","financialservices.gov.in",
    "commerce.gov.in","sportsauthorityofindia.nic.in","agriculture.gov.in","power.gov.in",
    "parliamentofindia.nic.in","taxindia.gov.in","cbic.gov.in","epfindia.gov.in","defence.gov.in",
    # International Government & UN/NGO
    ".gov",".europa.eu","un.org","who.int","nasa.gov","esa.int","imf.org","worldbank.org",
    "fao.org","wto.org","unicef.org","unhcr.org","redcross.org","cdc.gov","nih.gov","usa.gov",
    "canada.ca","gov.uk","australia.gov.au","japan.go.jp","ec.europa.eu","consilium.europa.eu",
    "ecb.europa.eu","unep.org","ilo.org","ohchr.org","unodc.org","unwomen.org",
    "unfpa.org","unesco.org","wmo.int","ifrc.org","nato.int","oecd.org","europarl.europa.eu",
    "unido.org","wfp.org"
]

# ==============================
# Helper Functions
# ==============================
def is_trusted(url):
    url = url.lower()
    return any(src in url for src in trusted_sources)

def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        article_div = soup.find("article") or soup.find("div", {"class": "articlebodycontent"}) or soup.find("div", {"id": "content-body"})
        if article_div:
            chunks = [elem.get_text().strip() for elem in article_div.find_all(["p","li","div"]) if len(elem.get_text().split())>5]
        else:
            chunks = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split())>5]
        text = " ".join(chunks)
        text = re.sub(r"\s+", " ", text)
        return text[:4000]
    except:
        return None

def query_gemini(text):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts":[{"text": f"Analyze the following news and give a reliable verdict (REAL or FAKE) with brief reasoning:\n\n{text}"}]}
        ]
    }
    try:
        resp = requests.post(GEMINI_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=20)
        if resp.status_code==200:
            data = resp.json()
            # Extract generated text
            parts = data.get("candidates", [{}])[0].get("content", [])
            combined_text = " ".join([p.get("text","") for p in parts])
            # Simple heuristic
            if "fake" in combined_text.lower():
                return "FAKE"
            elif "real" in combined_text.lower():
                return "REAL"
            else:
                return "UNSURE"
        else:
            return f"ERROR {resp.status_code}"
    except Exception as e:
        return f"Exception: {e}"

# ==============================
# Final Decision
# ==============================
def final_decision(text="", url=""):
    if url and is_trusted(url):
        return "REAL"
    if url and not is_trusted(url):
        scraped = scrape_url(url)
        if scraped:
            return query_gemini(scraped)
        else:
            return "UNSURE (Could not scrape URL)"
    if text:
        return query_gemini(text)
    return "UNSURE (No input provided)"

# ==============================
# Streamlit UI
# ==============================
st.title("📰 News Fact Checker (Gemini API)")

input_type = st.radio("Choose Input Type", ["Text", "URL"])
user_text = ""
page_url = ""

if input_type=="Text":
    user_text = st.text_area("Enter news text here:", height=250)
elif input_type=="URL":
    page_url = st.text_input("Enter news article URL:")

if st.button("Analyze"):
    if not user_text.strip() and not page_url.strip():
        st.warning("Please enter valid text or URL.")
    else:
        result = final_decision(user_text, page_url)
        if result=="REAL":
            st.success("🟢 REAL NEWS")
        elif result=="FAKE":
            st.error("🔴 FAKE NEWS")
        else:
            st.info(f"⚪ Verdict: {result}")
