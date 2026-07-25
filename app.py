import streamlit as st
import streamlit.components.v1 as components
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import spacy
import time

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NLP Toolkit - Creative Edition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS WITH YOUR COLOR PALETTE
# --------------------------------------------------

def load_css():
    st.markdown("""
    <style>
        /* ==========================================================
           Google Fonts
        ========================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700&display=swap');

        /* ==========================================================
           Color Variables
        ========================================================== */
        :root {
            --honeydew: #E5F8F0;
            --tea-green: #ECFFBE;
            --mauve: #BCA4F5;
            --sky-blue: #81CFFF;
            --royal-blue: #4A69CE;
        }

        /* ==========================================================
           Animated Background
        ========================================================== */
        .stApp {
            background: linear-gradient(135deg, #E5F8F0 0%, #ECFFBE 30%, #BCA4F5 70%, #81CFFF 100%);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            position: relative;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 80%, rgba(188, 164, 245, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(129, 207, 255, 0.15) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }

        header { visibility: hidden; }
        footer { visibility: hidden; }

        .block-container {
            position: relative;
            z-index: 1;
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* ==========================================================
           Floating Particles
        ========================================================== */
        .particles-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }

        /* ==========================================================
           Glass Morphism Header
        ========================================================== */
        .glass-header {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 30px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(74, 105, 206, 0.1);
            animation: floatIn 0.8s ease-out;
            position: relative;
            overflow: hidden;
        }

        .glass-header::before {
            content: '✦';
            position: absolute;
            top: -30px;
            right: 30px;
            font-size: 120px;
            color: rgba(188, 164, 245, 0.15);
            animation: spin 20s linear infinite;
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .glass-header::after {
            content: '✦';
            position: absolute;
            bottom: -40px;
            left: 20px;
            font-size: 150px;
            color: rgba(129, 207, 255, 0.12);
            animation: spin 25s linear infinite reverse;
        }

        .header-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.2rem;
            font-weight: 700;
            color: #4A69CE;
            margin: 0;
            text-align: center;
            position: relative;
            z-index: 1;
            animation: titleGlow 3s ease-in-out infinite;
        }

        @keyframes titleGlow {
            0%, 100% { text-shadow: 0 0 20px rgba(74, 105, 206, 0.1); }
            50% { text-shadow: 0 0 40px rgba(74, 105, 206, 0.2); }
        }

        .header-title span {
            background: linear-gradient(135deg, #4A69CE, #BCA4F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-subtitle {
            text-align: center;
            color: #4A69CE;
            font-size: 1.1rem;
            font-weight: 400;
            margin-top: 0.3rem;
            position: relative;
            z-index: 1;
            opacity: 0.8;
        }

        /* ==========================================================
           Glass Cards
        ========================================================== */
        .glass-card {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 1.8rem;
            border: 1px solid rgba(255, 255, 255, 0.25);
            box-shadow: 0 8px 32px rgba(74, 105, 206, 0.08);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            animation: cardFade 0.6s ease-out;
        }

        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 48px rgba(74, 105, 206, 0.15);
            border-color: rgba(255, 255, 255, 0.4);
        }

        @keyframes cardFade {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* ==========================================================
           Text Area - Glass Input
        ========================================================== */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.3) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            color: #4A69CE !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 20px !important;
            padding: 18px !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 16px rgba(74, 105, 206, 0.05) !important;
        }

        .stTextArea textarea:focus {
            border-color: #4A69CE !important;
            box-shadow: 0 0 0 4px rgba(74, 105, 206, 0.08), 0 8px 32px rgba(74, 105, 206, 0.1) !important;
            background: rgba(255, 255, 255, 0.4) !important;
        }

        .stTextArea textarea::placeholder {
            color: rgba(74, 105, 206, 0.4) !important;
        }

        /* ==========================================================
           Button - Creative Gradient
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 58px;
            border: none;
            border-radius: 30px;
            background: linear-gradient(135deg, #4A69CE 0%, #BCA4F5 50%, #81CFFF 100%);
            background-size: 200% 200%;
            color: white;
            font-size: 17px;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 24px rgba(74, 105, 206, 0.25);
            letter-spacing: 0.5px;
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
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.6s ease;
        }

        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 40px rgba(74, 105, 206, 0.35);
            background-position: 100% 100%;
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        .stButton > button:active {
            transform: translateY(0px) scale(0.98);
        }

        /* ==========================================================
           Metric Cards - Creative
        ========================================================== */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin: 0.5rem 0 1rem 0;
        }

        .metric-item {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1.5rem 1rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .metric-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #4A69CE, #BCA4F5, #81CFFF);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .metric-item:hover::before {
            opacity: 1;
        }

        .metric-item:hover {
            transform: translateY(-6px) scale(1.02);
            background: rgba(255, 255, 255, 0.3);
            box-shadow: 0 12px 40px rgba(74, 105, 206, 0.15);
        }

        .metric-icon { 
            font-size: 1.8rem; 
            margin-bottom: 4px;
            display: inline-block;
            animation: bounce 2s ease-in-out infinite;
        }

        .metric-item:nth-child(2) .metric-icon { animation-delay: 0.2s; }
        .metric-item:nth-child(3) .metric-icon { animation-delay: 0.4s; }
        .metric-item:nth-child(4) .metric-icon { animation-delay: 0.6s; }
        .metric-item:nth-child(5) .metric-icon { animation-delay: 0.8s; }
        .metric-item:nth-child(6) .metric-icon { animation-delay: 1s; }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }

        .metric-number {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #4A69CE, #BCA4F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 4px 0;
            line-height: 1.2;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #4A69CE;
            font-weight: 500;
            opacity: 0.8;
        }

        /* ==========================================================
           Tabs - Creative
        ========================================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 8px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 14px;
            padding: 0.7rem 1.4rem;
            font-weight: 500;
            font-size: 0.85rem;
            color: #4A69CE;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            background: transparent;
            border: none;
            opacity: 0.6;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.2);
            opacity: 1;
            transform: translateY(-2px);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(255, 255, 255, 0.3) !important;
            color: #4A69CE !important;
            box-shadow: 0 4px 16px rgba(74, 105, 206, 0.1);
            font-weight: 600;
            opacity: 1;
            backdrop-filter: blur(8px);
        }

        /* ==========================================================
           Result Boxes
        ========================================================== */
        .result-box {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 1rem 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            margin-bottom: 0.75rem;
            color: #4A69CE;
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.3s ease;
            border-left: 4px solid #BCA4F5;
        }

        .result-box:hover {
            background: rgba(255, 255, 255, 0.25);
            transform: translateX(6px);
            border-left-color: #4A69CE;
        }

        .token-box {
            display: inline-block;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-size: 0.9rem;
            color: #4A69CE;
            font-weight: 500;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .token-box:hover {
            background: rgba(74, 105, 206, 0.15);
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 4px 16px rgba(74, 105, 206, 0.15);
        }

        /* ==========================================================
           Quick Stats
        ========================================================== */
        .quick-stats {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1rem 1.25rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 4px 16px rgba(74, 105, 206, 0.05);
        }

        .quick-stats p {
            margin: 0.4rem 0;
            color: #4A69CE;
            font-size: 0.95rem;
        }

        .quick-stats strong {
            color: #4A69CE;
            font-weight: 700;
        }

        .common-words {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 0.75rem 1.25rem;
            border-radius: 16px;
            margin-top: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #4A69CE;
            font-size: 0.95rem;
        }

        .common-words strong {
            color: #4A69CE;
        }

        .common-words span {
            display: inline-block;
            background: rgba(255, 255, 255, 0.15);
            padding: 0.2rem 0.8rem;
            border-radius: 30px;
            margin: 0.2rem;
            transition: all 0.3s ease;
        }

        .common-words span:hover {
            background: rgba(74, 105, 206, 0.15);
            transform: scale(1.05);
        }

        /* ==========================================================
           Download Button
        ========================================================== */
        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #4A69CE, #BCA4F5) !important;
            border-radius: 30px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 20px rgba(74, 105, 206, 0.2) !important;
            height: 52px !important;
            font-size: 15px !important;
            color: white !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 32px rgba(74, 105, 206, 0.3) !important;
        }

        /* ==========================================================
           Alerts
        ========================================================== */
        [data-testid="stAlert"] {
            border-radius: 16px !important;
            border: none !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
        }

        /* ==========================================================
           Section Headers
        ========================================================== */
        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #4A69CE;
            margin: 1.5rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideIn 0.5s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .section-title-emoji { 
            font-size: 1.5rem; 
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        /* ==========================================================
           Footer
        ========================================================== */
        .footer {
            text-align: center;
            color: #4A69CE;
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            margin-top: 2rem;
            opacity: 0.7;
        }

        .footer span {
            font-weight: 600;
            background: linear-gradient(135deg, #4A69CE, #BCA4F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* ==========================================================
           Scrollbar
        ========================================================== */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #4A69CE, #BCA4F5);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: #4A69CE; }

        /* ==========================================================
           Responsive
        ========================================================== */
        @media (max-width: 768px) {
            .header-title { font-size: 2rem; }
            .glass-header { padding: 1.5rem; }
            .block-container { padding: 1rem !important; }
            .metric-grid { grid-template-columns: repeat(3, 1fr); gap: 10px; }
            .metric-number { font-size: 1.6rem; }
            .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.8rem; font-size: 0.75rem; }
        }

        /* ==========================================================
           Loading Animation
        ========================================================== */
        .custom-spinner {
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 4px solid rgba(74, 105, 206, 0.1);
            border-top: 4px solid #4A69CE;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# FLOATING PARTICLES (JavaScript)
# --------------------------------------------------

def add_particles():
    components.html("""
    <div id="particles-container" style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;"></div>
    <script>
        (function() {
            const container = document.getElementById('particles-container');
            const colors = ['#E5F8F0', '#ECFFBE', '#BCA4F5', '#81CFFF', '#4A69CE'];
            const particles = [];
            
            for (let i = 0; i < 40; i++) {
                const particle = document.createElement('div');
                const size = Math.random() * 6 + 3;
                particle.style.cssText = `
                    position: absolute;
                    width: ${size}px;
                    height: ${size}px;
                    background: ${colors[Math.floor(Math.random() * colors.length)]};
                    border-radius: 50%;
                    left: ${Math.random() * 100}%;
                    top: ${Math.random() * 100}%;
                    opacity: ${Math.random() * 0.3 + 0.1};
                    animation: floatParticle ${Math.random() * 20 + 15}s linear infinite;
                    animation-delay: ${Math.random() * 10}s;
                `;
                container.appendChild(particle);
                particles.push({
                    el: particle,
                    x: parseFloat(particle.style.left),
                    y: parseFloat(particle.style.top),
                    dx: (Math.random() - 0.5) * 0.02,
                    dy: (Math.random() - 0.5) * 0.02,
                });
            }
            
            const style = document.createElement('style');
            style.textContent = `
                @keyframes floatParticle {
                    0% { transform: translate(0, 0); opacity: 0.1; }
                    50% { opacity: 0.3; }
                    100% { transform: translate(${window.innerWidth * 0.1}px, ${window.innerHeight * 0.1}px); opacity: 0.1; }
                }
            `;
            document.head.appendChild(style);
        })();
    </script>
    """, height=0)

add_particles()

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
# HERO HEADER
# --------------------------------------------------

st.markdown("""
<div class="glass-header">
    <h1 class="header-title">🧠 <span>NLP Toolkit</span></h1>
    <p class="header-subtitle">✨ Advanced Natural Language Processing — Analyze, Understand, and Extract Insights</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown("""
<div class="section-title">
    <span class="section-title-emoji">✍️</span> Input Text
</div>
""", unsafe_allow_html=True)

text = st.text_area(
    "Enter your text for analysis",
    height=200,
    placeholder="Type or paste your English text here...\n\nExample: Apple Inc. is planning to open a new store in New York next month. The company's CEO, Tim Cook, announced this exciting news yesterday.",
    label_visibility="collapsed"
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze = st.button("✨ Analyze Text", use_container_width=True)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
        st.stop()
    
    with st.spinner("🧠 Analyzing your text..."):
        time.sleep(0.5)
        
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
        st.markdown("""
        <div class="section-title">
            <span class="section-title-emoji">📄</span> Original Text
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-title">
            <span class="section-title-emoji">📊</span> Quick Stats
        </div>
        """, unsafe_allow_html=True)
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
    st.markdown("""
    <div class="section-title">
        <span class="section-title-emoji">📈</span> Analysis Metrics
    </div>
    """, unsafe_allow_html=True)
    
    metrics_data = [
        ("📝", "Total Words", len(words)),
        ("📑", "Sentences", len(sentences)),
        ("🧹", "After Stopwords", len(filtered_words)),
        ("🏷️", "Named Entities", len(entities)),
        ("🔗", "Noun Phrases", len(chunks)),
        ("✨", "Unique Words", unique_words),
    ]
    
    # Create metric cards with animation
    metrics_html = '<div class="metric-grid">'
    for icon, label, value in metrics_data:
        metrics_html += f"""
        <div class="metric-item" data-value="{value}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-number" data-target="{value}">0</div>
            <div class="metric-label">{label}</div>
        </div>
        """
    metrics_html += '</div>'
    
    metrics_html += """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const items = document.querySelectorAll('.metric-item');
        items.forEach((item) => {
            const target = parseInt(item.dataset.value) || 0;
            const numEl = item.querySelector('.metric-number');
            const duration = 1200;
            const startTime = performance.now();
            
            function animate(ts) {
                const progress = Math.min((ts - startTime) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                numEl.textContent = Math.round(eased * target);
                if (progress < 1) requestAnimationFrame(animate);
            }
            requestAnimationFrame(animate);
        });
    });
    </script>
    """
    
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Most common words
    if most_common:
        common_words_html = '<div class="common-words"><strong>🔥 Most Common Words:</strong> '
        for word, count in most_common:
            common_words_html += f'<span>{word} ({count})</span>'
        common_words_html += '</div>'
        st.markdown(common_words_html, unsafe_allow_html=True)
    
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
    
    st.success("✨ NLP Analysis Completed Successfully!")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, NLTK & spaCy • 
    <span>NLP Toolkit</span>
</div>
""", unsafe_allow_html=True)