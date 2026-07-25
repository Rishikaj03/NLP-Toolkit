import streamlit as st
import streamlit.components.v1 as components
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS FOR MODERN UI
# --------------------------------------------------
def load_css():
    st.markdown("""
    <style>
        /* ==========================================================
           Google Fonts
        ========================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ==========================================================
           Reset & Base
        ========================================================== */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 15px;
            line-height: 1.6;
        }

        /* ==========================================================
           Background - Clean gradient
        ========================================================== */
        .stApp {
            background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        }

        header { visibility: hidden; }
        footer { visibility: hidden; }

        .block-container {
            max-width: 1300px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* ==========================================================
           Sidebar - Clean and minimal
        ========================================================== */
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5E7EB;
            padding: 1.5rem 1rem;
        }

        section[data-testid="stSidebar"] .sidebar-content {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #F3F4F6;
        }

        .sidebar-logo {
            font-size: 2rem;
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
        }

        .sidebar-title {
            font-weight: 700;
            font-size: 1.2rem;
            color: #1F2937;
            letter-spacing: -0.3px;
        }

        .sidebar-subtitle {
            font-size: 0.8rem;
            color: #6B7280;
            margin-top: -2px;
        }

        .sidebar-section {
            margin: 1.5rem 0 1rem 0;
        }

        .sidebar-section-title {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #9CA3AF;
            margin-bottom: 0.75rem;
        }

        .tech-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 4px;
        }

        .tech-tag {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            background: #F3F4F6;
            color: #4B5563;
            border: 1px solid #E5E7EB;
        }

        .tech-tag.primary { background: #EEF2FF; color: #4F46E5; border-color: #C7D2FE; }
        .tech-tag.pink { background: #FDF2F8; color: #DB2777; border-color: #FBCFE8; }
        .tech-tag.cyan { background: #ECFEFF; color: #0891B2; border-color: #A5F3FC; }
        .tech-tag.purple { background: #F5F3FF; color: #7C3AED; border-color: #DDD6FE; }

        .sidebar-footer {
            margin-top: auto;
            padding-top: 1rem;
            border-top: 1px solid #F3F4F6;
            font-size: 0.75rem;
            color: #9CA3AF;
        }

        /* ==========================================================
           Header Hero
        ========================================================== */
        .hero-container {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(79, 70, 229, 0.15);
        }

        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 400px;
            height: 400px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 50%;
            pointer-events: none;
        }

        .hero-container::after {
            content: '';
            position: absolute;
            bottom: -40%;
            left: -10%;
            width: 300px;
            height: 300px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 50%;
            pointer-events: none;
        }

        .hero-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0;
            position: relative;
            z-index: 1;
        }

        .hero-title span {
            background: linear-gradient(135deg, #FDE68A, #FCD34D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-subtitle {
            color: rgba(255, 255, 255, 0.85);
            font-size: 1.05rem;
            margin-top: 0.5rem;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }

        /* ==========================================================
           Section Headers
        ========================================================== */
        .section-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1F2937;
            margin: 1.5rem 0 1rem 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-title-emoji { font-size: 1.4rem; }

        /* ==========================================================
           Cards
        ========================================================== */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid #F3F4F6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            margin-bottom: 1.5rem;
            transition: all 0.25s ease;
        }

        .card:hover {
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border-color: #E5E7EB;
        }

        /* ==========================================================
           Text Area
        ========================================================== */
        .stTextArea textarea {
            background: white !important;
            color: #1F2937 !important;
            border: 2px solid #E5E7EB !important;
            border-radius: 14px !important;
            padding: 16px !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        }

        .stTextArea textarea:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.08), 0 4px 12px rgba(99, 102, 241, 0.04) !important;
        }

        .stTextArea textarea::placeholder {
            color: #9CA3AF !important;
        }

        /* ==========================================================
           Button
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 54px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            color: white;
            font-size: 16px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25);
            letter-spacing: 0.3px;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
        }

        .stButton > button:active {
            transform: translateY(0px);
        }

        /* ==========================================================
           Metric Cards
        ========================================================== */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 14px;
            margin: 0.5rem 0 1rem 0;
        }

        .metric-item {
            background: white;
            padding: 1.25rem 1rem;
            border-radius: 14px;
            border: 1px solid #F3F4F6;
            text-align: center;
            transition: all 0.25s ease;
        }

        .metric-item:hover {
            transform: translateY(-4px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border-color: #E5E7EB;
        }

        .metric-icon { font-size: 1.5rem; margin-bottom: 4px; }

        .metric-number {
            font-size: 2rem;
            font-weight: 800;
            color: #1F2937;
            margin: 4px 0;
            line-height: 1.2;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #6B7280;
            font-weight: 500;
        }

        /* ==========================================================
           Tabs
        ========================================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #F9FAFB;
            padding: 6px;
            border-radius: 14px;
            border: 1px solid #F3F4F6;
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-size: 0.85rem;
            color: #6B7280;
            font-family: 'Inter', sans-serif;
            transition: all 0.2s ease;
            background: transparent;
            border: none;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.7);
            color: #1F2937;
        }

        .stTabs [aria-selected="true"] {
            background: white !important;
            color: #6366F1 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            font-weight: 600;
        }

        /* ==========================================================
           Result Boxes
        ========================================================== */
        .result-box {
            background: #F9FAFB;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border-left: 4px solid #6366F1;
            border-top: 1px solid #F3F4F6;
            border-right: 1px solid #F3F4F6;
            border-bottom: 1px solid #F3F4F6;
            margin-bottom: 0.75rem;
            color: #1F2937;
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.2s ease;
        }

        .result-box:hover {
            background: white;
            border-left-color: #8B5CF6;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.06);
        }

        .token-box {
            display: inline-block;
            background: #F3F4F6;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            font-size: 0.9rem;
            color: #1F2937;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .token-box:hover {
            background: #EEF2FF;
            border-color: #C7D2FE;
            transform: translateY(-2px);
        }

        /* ==========================================================
           Quick Stats
        ========================================================== */
        .quick-stats {
            background: white;
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border: 1px solid #F3F4F6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }

        .quick-stats p {
            margin: 0.4rem 0;
            color: #4B5563;
            font-size: 0.95rem;
        }

        .quick-stats strong {
            color: #1F2937;
            font-weight: 600;
        }

        .common-words {
            background: white;
            padding: 0.75rem 1.25rem;
            border-radius: 12px;
            margin-top: 0.75rem;
            border: 1px solid #F3F4F6;
            color: #1F2937;
            font-size: 0.95rem;
        }

        .common-words strong {
            color: #6366F1;
        }

        /* ==========================================================
           DataFrames
        ========================================================== */
        [data-testid="stDataFrame"] {
            border-radius: 12px !important;
            overflow: hidden !important;
            border: 1px solid #F3F4F6 !important;
        }

        /* ==========================================================
           Download Button
        ========================================================== */
        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2) !important;
            height: 50px !important;
            font-size: 15px !important;
            color: white !important;
            border: none !important;
        }

        [data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3) !important;
        }

        /* ==========================================================
           Alerts
        ========================================================== */
        [data-testid="stAlert"] {
            border-radius: 12px !important;
            border: none !important;
        }

        /* ==========================================================
           Footer
        ========================================================== */
        .footer {
            text-align: center;
            color: #9CA3AF;
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid #F3F4F6;
            margin-top: 2rem;
        }

        .footer span {
            color: #6366F1;
            font-weight: 600;
        }

        /* ==========================================================
           Scrollbar
        ========================================================== */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #F3F4F6; border-radius: 10px; }
        ::-webkit-scrollbar-thumb {
            background: #D1D5DB;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

        /* ==========================================================
           Responsive
        ========================================================== */
        @media (max-width: 768px) {
            .hero-title { font-size: 1.8rem; }
            .hero-container { padding: 1.5rem; }
            .block-container { padding: 1rem !important; }
            .metric-grid { grid-template-columns: repeat(3, 1fr); gap: 10px; }
            .metric-number { font-size: 1.5rem; }
            .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.8rem; font-size: 0.75rem; }
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
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">🧠</div>
        <div>
            <div class="sidebar-title">NLP Toolkit</div>
            <div class="sidebar-subtitle">v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Tech Stack</div>
        <div class="tech-tags">
            <span class="tech-tag primary">Python</span>
            <span class="tech-tag pink">NLTK</span>
            <span class="tech-tag purple">spaCy</span>
            <span class="tech-tag cyan">Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-footer">
        ⚡ Natural Language Processing made simple
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🧠 Natural Language <span>Processing</span></h1>
    <p class="hero-subtitle">Analyze, understand, and extract insights from your text with advanced NLP techniques</p>
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
    analyze = st.button("🚀 Analyze Text", use_container_width=True)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
        st.stop()
    
    with st.spinner("🔄 Processing your text..."):
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
    
    # Create metric cards using HTML/CSS with animation
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
        items.forEach((item, index) => {
            const target = parseInt(item.dataset.value) || 0;
            const numEl = item.querySelector('.metric-number');
            const duration = 1000;
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
    <span>Natural Language Processing Toolkit</span>
</div>
""", unsafe_allow_html=True)