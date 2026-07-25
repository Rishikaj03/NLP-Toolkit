import streamlit as st
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import spacy

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Natural Language Processing Toolkit",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS FOR PROFESSIONAL UI
# --------------------------------------------------

def load_css():
    st.markdown("""
    <style>
        /* ==========================================================
           Google Fonts
        ========================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        /* ==========================================================
           Global
        ========================================================== */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        :root {
            --grad-a: #7C3AED;
            --grad-b: #EC4899;
            --grad-c: #06B6D4;
            --grad-d: #F59E0B;
        }

        /* ==========================================================
           Animated Background
        ========================================================== */
        .stApp {
            background: linear-gradient(-45deg, #0F0C29, #302B63, #24243e, #1a1a2e);
            background-size: 400% 400%;
            animation: gradientShift 18s ease infinite;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* floating orbs behind content */
        .stApp::before, .stApp::after {
            content: '';
            position: fixed;
            border-radius: 50%;
            filter: blur(90px);
            z-index: 0;
            pointer-events: none;
        }
        .stApp::before {
            width: 420px; height: 420px;
            top: -120px; left: -100px;
            background: radial-gradient(circle, rgba(124,58,237,0.35), transparent 70%);
            animation: floatOrb1 14s ease-in-out infinite;
        }
        .stApp::after {
            width: 480px; height: 480px;
            bottom: -140px; right: -120px;
            background: radial-gradient(circle, rgba(6,182,212,0.30), transparent 70%);
            animation: floatOrb2 16s ease-in-out infinite;
        }
        @keyframes floatOrb1 {
            0%, 100% { transform: translate(0,0) scale(1); }
            50% { transform: translate(40px, 60px) scale(1.15); }
        }
        @keyframes floatOrb2 {
            0%, 100% { transform: translate(0,0) scale(1); }
            50% { transform: translate(-50px, -30px) scale(1.1); }
        }

        /* ==========================================================
           Hide Header/Footer
        ========================================================== */
        header { visibility: hidden; }
        footer { visibility: hidden; }

        /* ==========================================================
           Main Container
        ========================================================== */
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            position: relative;
            z-index: 1;
        }

        /* ==========================================================
           Custom Header - Vivid animated gradient + glow border
        ========================================================== */
        .header-container {
            background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 45%, #831843 100%);
            padding: 3rem 2rem 2.4rem 2rem;
            border-radius: 26px;
            margin-bottom: 2.5rem;
            box-shadow: 0 12px 40px rgba(124, 58, 237, 0.35), 0 0 0 1px rgba(255,255,255,0.08) inset;
            position: relative;
            overflow: hidden;
            animation: headerGlow 6s ease-in-out infinite;
        }

        @keyframes headerGlow {
            0%, 100% { box-shadow: 0 12px 40px rgba(124, 58, 237, 0.35), 0 0 0 1px rgba(255,255,255,0.08) inset; }
            50% { box-shadow: 0 12px 55px rgba(236, 72, 153, 0.4), 0 0 0 1px rgba(255,255,255,0.1) inset; }
        }

        .header-container::before {
            content: '';
            position: absolute;
            top: -60%;
            right: -15%;
            width: 420px;
            height: 420px;
            background: radial-gradient(circle, rgba(96, 165, 250, 0.20) 0%, transparent 70%);
            border-radius: 50%;
            animation: spin 20s linear infinite;
        }

        .header-container::after {
            content: '';
            position: absolute;
            bottom: -40%;
            left: -12%;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(236, 72, 153, 0.18) 0%, transparent 70%);
            border-radius: 50%;
            animation: spin 25s linear infinite reverse;
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .header-title {
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            color: #FFFFFF;
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 20px rgba(0,0,0,0.25);
        }

        .header-title span {
            background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6, #FBBF24);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shimmerText 6s ease infinite;
        }

        @keyframes shimmerText {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        .header-subtitle {
            color: rgba(255, 255, 255, 0.78);
            text-align: center;
            font-size: 1.1rem;
            margin-top: 0.6rem;
            font-weight: 400;
            letter-spacing: 0.3px;
            position: relative;
            z-index: 1;
        }

        /* ==========================================================
           Card Styling - Glass morphism with vivid hover
        ========================================================== */
        .card {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(14px);
            padding: 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            margin-bottom: 1.5rem;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .card:hover {
            box-shadow: 0 10px 32px rgba(124, 58, 237, 0.25);
            border-color: rgba(167, 139, 250, 0.4);
            transform: translateY(-3px);
        }

        /* ==========================================================
           Headings inside content
        ========================================================== */
        h1, h2, h3 {
            color: #F1F5F9 !important;
            font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        }

        .stCaption, [data-testid="stCaptionContainer"] {
            color: rgba(226, 232, 240, 0.65) !important;
        }

        /* ==========================================================
           Text Area - Glowing focus ring
        ========================================================== */
        .stTextArea textarea {
            background: rgba(15, 23, 42, 0.55) !important;
            color: #F1F5F9 !important;
            border: 2px solid rgba(148, 163, 184, 0.25) !important;
            border-radius: 16px !important;
            padding: 18px !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }

        .stTextArea textarea:focus {
            border-color: #A78BFA !important;
            box-shadow: 0 0 0 4px rgba(167, 139, 250, 0.18), 0 4px 24px rgba(167, 139, 250, 0.15) !important;
        }

        .stTextArea textarea::placeholder {
            color: rgba(148, 163, 184, 0.7) !important;
            font-weight: 300;
        }

        /* ==========================================================
           Button - Punchy animated gradient
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 56px;
            border: none;
            border-radius: 16px;
            background: linear-gradient(135deg, #7C3AED 0%, #EC4899 50%, #F59E0B 100%);
            background-size: 200% 200%;
            color: white;
            font-size: 17px;
            font-weight: 700;
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 6px 24px rgba(124, 58, 237, 0.4);
            letter-spacing: 0.4px;
            position: relative;
            overflow: hidden;
        }

        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
            transition: left 0.6s ease;
        }

        .stButton > button:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 10px 36px rgba(236, 72, 153, 0.45);
            background-position: 100% 100%;
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        .stButton > button:active {
            transform: translateY(-1px) scale(0.99);
        }

        /* ==========================================================
           Metric Cards - Neon accent + lift
        ========================================================== */
        .metric-card {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(10px);
            padding: 22px 16px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            text-align: center;
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 16px rgba(0,0,0,0.18);
            position: relative;
            overflow: hidden;
            margin-bottom: 18px;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #7C3AED, #EC4899, #F59E0B, #06B6D4);
            background-size: 300% 100%;
            opacity: 0;
            transition: opacity 0.3s ease;
            animation: borderFlow 4s linear infinite;
        }

        @keyframes borderFlow {
            0% { background-position: 0% 0%; }
            100% { background-position: 300% 0%; }
        }

        .metric-card:hover::before {
            opacity: 1;
        }

        .metric-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 12px 32px rgba(124, 58, 237, 0.3);
            border-color: rgba(167, 139, 250, 0.5);
        }

        .metric-number {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #A78BFA 0%, #F472B6 50%, #FBBF24 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.2;
        }

        .metric-label {
            font-size: 0.85rem;
            color: rgba(226, 232, 240, 0.7);
            margin-top: 0.35rem;
            font-weight: 500;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }

        /* ==========================================================
           Tabs - Glowing active state
        ========================================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 6px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 14px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-size: 0.9rem;
            color: rgba(226, 232, 240, 0.65);
            font-family: 'Inter', sans-serif;
            transition: all 0.25s ease;
            background: transparent;
            border: none;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.08);
            color: #F1F5F9;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #7C3AED, #EC4899) !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 18px rgba(236, 72, 153, 0.4);
            font-weight: 600;
        }

        /* ==========================================================
           Result Boxes - Gradient border, subtle glow on hover
        ========================================================== */
        .result-box {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(6px);
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border-left: 4px solid transparent;
            background-image: linear-gradient(rgba(15,23,42,0.55), rgba(15,23,42,0.55)), linear-gradient(135deg, #7C3AED, #EC4899);
            background-origin: padding-box, border-box;
            background-clip: padding-box, border-box;
            margin-bottom: 0.75rem;
            color: #F1F5F9;
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.25s ease;
        }

        .result-box:hover {
            background-image: linear-gradient(rgba(15,23,42,0.7), rgba(15,23,42,0.7)), linear-gradient(135deg, #A78BFA, #F472B6);
            transform: translateX(6px);
            box-shadow: 0 4px 20px rgba(124, 58, 237, 0.2);
        }

        .token-box {
            display: inline-block;
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(4px);
            padding: 0.35rem 0.85rem;
            margin: 0.22rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            font-size: 0.9rem;
            color: #F1F5F9;
            font-weight: 500;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .token-box:hover {
            background: linear-gradient(135deg, #7C3AED, #EC4899);
            border-color: transparent;
            color: #FFFFFF;
            transform: translateY(-2px) scale(1.06);
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4);
        }

        /* ==========================================================
           Quick Stats - Glass card
        ========================================================== */
        .quick-stats {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 1rem 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }

        .quick-stats p {
            margin: 0.4rem 0;
            color: rgba(226, 232, 240, 0.85);
            font-size: 0.95rem;
        }

        .quick-stats strong {
            background: linear-gradient(135deg, #A78BFA, #F472B6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
        }

        /* ==========================================================
           Common Words Box
        ========================================================== */
        .common-words {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(6px);
            padding: 0.85rem 1.25rem;
            border-radius: 14px;
            margin-top: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #F1F5F9;
            font-size: 0.95rem;
        }

        .common-words strong {
            background: linear-gradient(135deg, #FBBF24, #F472B6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* ==========================================================
           Dataframes
        ========================================================== */
        [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.12);
        }

        /* ==========================================================
           Footer
        ========================================================== */
        .footer {
            text-align: center;
            color: rgba(148, 163, 184, 0.7);
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            margin-top: 2rem;
        }

        .footer span {
            background: linear-gradient(135deg, #A78BFA, #F472B6, #FBBF24);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
        }

        /* ==========================================================
           Scrollbar
        ========================================================== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #7C3AED, #EC4899);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #A78BFA, #F472B6);
        }

        /* ==========================================================
           Alerts (success/warning/info) restyled
        ========================================================== */
        [data-testid="stAlert"] {
            border-radius: 14px !important;
            backdrop-filter: blur(8px);
        }

        /* ==========================================================
           Download button accent
        ========================================================== */
        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #06B6D4 0%, #7C3AED 100%) !important;
            box-shadow: 0 6px 24px rgba(6, 182, 212, 0.35) !important;
        }

        /* ==========================================================
           Responsive
        ========================================================== */
        @media (max-width: 768px) {
            .header-title {
                font-size: 2rem;
            }

            .metric-number {
                font-size: 1.8rem;
            }

            .stTabs [data-baseweb="tab"] {
                padding: 0.4rem 0.8rem;
                font-size: 0.8rem;
            }
        }

        /* Mobile Responsive */
        @media (max-width: 768px){
            h1,h2,h3{ color:#F1F5F9;}

            h1{
                font-size:2rem !important;
                text-align:center;
            }

            h2{
                font-size:1.5rem !important;
            }

            h3{
                font-size:1.2rem !important;
            }

            .block-container{
                padding:1rem !important;
            }

            .stTextArea textarea{
                min-height:180px !important;
                font-size:16px !important;
            }

            .stButton button{
                width:100%;
                font-size:18px;
                height:55px;
            }

            div[data-testid="stMetric"]{
                margin-bottom:12px;
            }
        }

    </style>
    """, unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# CACHE RESOURCES
# --------------------------------------------------

@st.cache_resource
def download_nltk():
    packages = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng"
    ]

    for package in packages:
        nltk.download(package, quiet=True)

download_nltk()

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="header-container">
    <h1 class="header-title">🧠 <span>NLP Toolkit</span></h1>
    <p class="header-subtitle">Advanced Natural Language Processing — Analyze, Understand, and Extract Insights</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

text = st.text_area(
    "📝 Enter your text for analysis",
    height=200,
    placeholder="Type or paste your English text here...\n\nExample: Apple Inc. is planning to open a new store in New York next month. The company's CEO, Tim Cook, announced this exciting news yesterday."
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze = st.button("🚀 Analyze Text", use_container_width=True)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
        st.stop()
    
    with st.spinner("🧠 Analyzing your text..."):
        # Sentence Segmentation
        sentences = sent_tokenize(text)
        
        # Word Tokenization
        words = word_tokenize(text)
        
        # Stopword Removal
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        
        # POS Tagging
        pos_tags = nltk.pos_tag(words)
        
        # NER
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Dependency Parsing
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        
        # Chunking
        chunks = [chunk.text for chunk in doc.noun_chunks]
        
        # Additional stats
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        unique_words = len(set(words))
        char_count = len(text)
        
        # Most common words
        word_freq = Counter([word.lower() for word in words if word.isalpha()])
        most_common = word_freq.most_common(5)
    
    # --------------------------------------------------
    # ORIGINAL TEXT
    # --------------------------------------------------
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📄 Original Text")
        st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        st.markdown(f"""
        <div class="quick-stats">
            <p><strong>Characters:</strong> {char_count:,}</p>
            <p><strong>Words:</strong> {len(words):,}</p>
            <p><strong>Sentences:</strong> {len(sentences):,}</p>
            <p><strong>Avg. Word Length:</strong> {avg_word_length:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{len(words)}</p>
        <p class="metric-label">Total Words</p>
    </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{len(sentences)}</p>
        <p class="metric-label">Sentences</p>
    </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{len(filtered_words)}</p>
        <p class="metric-label">After Stopwords</p>
    </div>
    """, unsafe_allow_html=True)
    
    col4.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{len(entities)}</p>
        <p class="metric-label">Named Entities</p>
    </div>
    """, unsafe_allow_html=True)
    
    col5.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{len(chunks)}</p>
        <p class="metric-label">Noun Phrases</p>
    </div>
    """, unsafe_allow_html=True)
    
    col6.markdown(f"""
    <div class="metric-card">
        <p class="metric-number">{unique_words}</p>
        <p class="metric-label">Unique Words</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Most common words
    if most_common:
        st.markdown(f"""
        <div class="common-words">
            <strong>🔥 Most Common Words:</strong> 
            {', '.join([f'"{word}" ({count})' for word, count in most_common])}
        </div>
        """, unsafe_allow_html=True)
    
    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📑 Sentences",
        "🔤 Tokens",
        "🚫 Stop Words",
        "🌱 Stemming",
        "📖 Lemmatization",
        "🏷️ POS Tags",
        "👤 NER",
        "🔗 Dependency",
        "📦 Chunking"
    ])
    
    # TAB 1 - Sentences
    with tab1:
        st.subheader("📑 Sentence Segmentation")
        st.caption("Breaking text into individual sentences")
        for i, sent in enumerate(sentences, 1):
            st.markdown(f'<div class="result-box"><strong>{i}.</strong> {sent}</div>', unsafe_allow_html=True)
    
    # TAB 2 - Tokens
    with tab2:
        st.subheader("🔤 Word Tokenization")
        st.caption("Breaking text into individual words/tokens")
        tokens_html = " ".join([f'<span class="token-box">{token}</span>' for token in words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
        st.caption(f"Total tokens: {len(words)}")
    
    # TAB 3 - Stop Words
    with tab3:
        st.subheader("🚫 Stop Word Removal")
        st.caption("Common words removed to focus on meaningful content")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in filtered_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
        st.caption(f"Tokens after stopword removal: {len(filtered_words)} (removed {len(words) - len(filtered_words)} stopwords)")
    
    # TAB 4 - Stemming
    with tab4:
        st.subheader("🌱 Stemming")
        st.caption("Reducing words to their root form using Porter Stemmer")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in stemmed_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
    
    # TAB 5 - Lemmatization
    with tab5:
        st.subheader("📖 Lemmatization")
        st.caption("Reducing words to their dictionary form using WordNet")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in lemmatized_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
    
    # TAB 6 - POS Tagging
    with tab6:
        st.subheader("🏷️ Part-of-Speech Tagging")
        st.caption("Grammatical tags assigned to each word")
        
        pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
        st.dataframe(
            pos_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Word": st.column_config.TextColumn("Word"),
                "POS Tag": st.column_config.TextColumn("POS Tag", help="Part of Speech tag")
            }
        )
    
    # TAB 7 - NER
    with tab7:
        st.subheader("👤 Named Entity Recognition")
        st.caption("Identifying named entities in text (people, organizations, locations, etc.)")
        
        if entities:
            ner_df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.dataframe(
                ner_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Entity": st.column_config.TextColumn("Entity"),
                    "Label": st.column_config.TextColumn("Entity Type")
                }
            )
        else:
            st.info("ℹ️ No named entities found in the text.")
    
    # TAB 8 - Dependency Parsing
    with tab8:
        st.subheader("🔗 Dependency Parsing")
        st.caption("Grammatical relationships between words in the sentence")
        
        dep_df = pd.DataFrame(dependencies, columns=["Word", "Dependency", "Head"])
        st.dataframe(
            dep_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Word": st.column_config.TextColumn("Word"),
                "Dependency": st.column_config.TextColumn("Dependency Relation"),
                "Head": st.column_config.TextColumn("Head Word")
            }
        )
    
    # TAB 9 - Chunking
    with tab9:
        st.subheader("📦 Noun Phrase Chunking")
        st.caption("Extracting noun phrases from the text")
        
        if chunks:
            for chunk in chunks:
                st.markdown(f'<div class="result-box">✅ {chunk}</div>', unsafe_allow_html=True)
            st.caption(f"Total noun phrases found: {len(chunks)}")
        else:
            st.info("ℹ️ No noun phrases found in the text.")
    
    # --------------------------------------------------
    # DOWNLOAD SECTION
    # --------------------------------------------------
    
    st.markdown("---")
    
    result = f"""
====================================
NATURAL LANGUAGE PROCESSING REPORT
====================================

Original Text:
{text}

------------------------------------
Analysis Summary
------------------------------------
Total Words: {len(words)}
Total Sentences: {len(sentences)}
Unique Words: {unique_words}
Average Word Length: {avg_word_length:.1f}
Named Entities: {len(entities)}
Noun Phrases: {len(chunks)}

------------------------------------
Sentences:
------------------------------------
{chr(10).join([f"{i+1}. {sent}" for i, sent in enumerate(sentences)])}

------------------------------------
Word Tokens:
------------------------------------
{', '.join(words)}

------------------------------------
Stop Word Removal:
------------------------------------
{', '.join(filtered_words)}

------------------------------------
Stemming:
------------------------------------
{', '.join(stemmed_words)}

------------------------------------
Lemmatization:
------------------------------------
{', '.join(lemmatized_words)}

------------------------------------
POS Tagging:
------------------------------------
{chr(10).join([f"{word}: {tag}" for word, tag in pos_tags])}

------------------------------------
Named Entity Recognition:
------------------------------------
{chr(10).join([f"{entity}: {label}" for entity, label in entities]) if entities else "No entities found"}

------------------------------------
Dependency Parsing:
------------------------------------
{chr(10).join([f"{word} -> {dep} -> {head}" for word, dep, head in dependencies])}

------------------------------------
Noun Phrase Chunking:
------------------------------------
{chr(10).join(chunks) if chunks else "No noun phrases found"}

====================================
Analysis completed successfully!
====================================
"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download Full Report",
            data=result,
            file_name="nlp_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.success("✅ NLP Analysis Completed Successfully!")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, NLTK & spaCy • 
    <span>Professional NLP Toolkit</span>
</div>
""", unsafe_allow_html=True)