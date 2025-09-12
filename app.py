import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
import json

# ==============================
# Gemini API function
# ==============================
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def query_gemini(text):
    API_KEY = "AIzaSyDlnSBUgoN2m94xmaFY2WIT-GjYC8MOUUg"  # hidden in code
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts":[{"text": f"Analyze the following news and give a reliable verdict (REAL or FAKE) with brief reasoning:\n\n{text}"}]}
        ]
    }
    try:
        resp = requests.post(f"{GEMINI_ENDPOINT}?key={API_KEY}", headers=headers, data=json.dumps(payload), timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            combined_text = ""

            for cand in candidates:
                if isinstance(cand, str):
                    combined_text += cand + " "
                elif isinstance(cand, dict):
                    content = cand.get("content", [])
                    for part in content:
                        combined_text += part.get("text","") + " "

            combined_text = combined_text.strip()

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
# Text cleaning
# ==============================
def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# Web scraping
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

# ==============================
# Trusted sources (200+)
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

def is_trusted(url):
    url = url.lower()
    return any(src in url for src in trusted_sources)

# ==============================
# Final decision
# ==============================
def final_decision(text, url=""):
    if url and is_trusted(url):
        return "REAL"
    return query_gemini(text)

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
            result = final_decision(user_input, page_url)
            if result=="REAL":
                st.success("🟢 REAL NEWS")
            elif result=="FAKE":
                st.error("🔴 FAKE NEWS")
            else:
                st.warning("⚠️ UNSURE / Could not classify confidently")
            with st.expander("📄 Extracted Text"):
                st.write(user_input)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
