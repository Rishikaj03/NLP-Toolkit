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
    page_title="NLP Toolkit - Text Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS WITH CREATIVE COLORS
# --------------------------------------------------
def load_css():
    st.markdown("""
    <style>
        /* ==========================================================
           Google Fonts
        ========================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        /* ==========================================================
           Color Variables
        ========================================================== */
        :root {
            --mauve: #BCA4F5;
            --mauve-dark: #A084E0;
            --sky-blue: #81CFFF;
            --royal-blue: #4A69CE;
            --royal-blue-dark: #3A52A8;
            --honeydew: #E5F8F0;
            --tea-green: #ECFFBE;
        }

        /* ==========================================================
           Base
        ========================================================== */
        html, body, [class*="css"] {
            font-family: 'Inter', 'sans serif', sans-serif;
            font-size: 15px;
            line-height: 1.6;
            color: #1F2937;
        }

        /* ==========================================================
           Background
        ========================================================== */
        .stApp {
            background: #F4F7FC;
            position: relative;
        }

        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 80%, rgba(188, 164, 245, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(129, 207, 255, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(74, 105, 206, 0.04) 0%, transparent 50%);
            z-index: 0;
            pointer-events: none;
        }

        header { visibility: hidden; }
        footer { visibility: hidden; }

        .block-container {
            max-width: 1300px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            position: relative;
            z-index: 1;
        }

        /* ==========================================================
           Sidebar
        ========================================================== */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid rgba(188, 164, 245, 0.15) !important;
            padding: 1.5rem 1rem;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid rgba(188, 164, 245, 0.15);
        }

        .sidebar-logo {
            font-size: 2rem;
            background: linear-gradient(135deg, #4F46E5, #BCA4F5);
            width: 48px;
            height: 48px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
        }

        .sidebar-title {
            font-weight: 700;
            font-size: 1.2rem;
            background: linear-gradient(135deg, #4F46E5, #BCA4F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .sidebar-subtitle {
            font-size: 0.8rem;
            color: #6B7280;
        }

        .sidebar-section-title {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #4F46E5;
            margin-bottom: 0.75rem;
        }

        .tech-tag {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            background: white;
            color: #4F46E5;
            border: 1px solid rgba(188, 164, 245, 0.2);
            transition: all 0.3s ease;
        }

        .tech-tag:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12);
            border-color: #4F46E5;
        }

        .tech-tag.primary { background: rgba(79, 70, 229, 0.08); color: #4F46E5; }
        .tech-tag.pink { background: rgba(188, 164, 245, 0.08); color: #8B5CF6; }
        .tech-tag.cyan { background: rgba(129, 207, 255, 0.08); color: #3A8BC0; }
        .tech-tag.purple { background: rgba(188, 164, 245, 0.08); color: #8B5CF6; }

        .sidebar-footer {
            margin-top: auto;
            padding-top: 1rem;
            border-top: 1px solid rgba(188, 164, 245, 0.12);
            font-size: 0.75rem;
            color: #9CA3AF;
        }

        /* ==========================================================
           Hero
        ========================================================== */
        .hero-container {
            background: linear-gradient(135deg, #4F46E5 0%, #BCA4F5 50%, #81CFFF 100%);
            padding: 3rem 2.5rem 2.5rem 2.5rem;
            border-radius: 24px;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 15px 50px rgba(79, 70, 229, 0.15);
        }

        .hero-title {
            color: white;
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }

        .hero-title span {
            background: linear-gradient(135deg, #E5F8F0, #ECFFBE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            margin-top: 0.5rem;
            font-weight: 300;
            position: relative;
            z-index: 1;
            letter-spacing: 0.5px;
        }

        /* ==========================================================
           Text Area
        ========================================================== */
        .stTextArea textarea {
            background: #FFFFFF !important;
            color: #1F2937 !important;
            border: 2px solid rgba(188, 164, 245, 0.2) !important;
            border-radius: 16px !important;
            padding: 18px !important;
            font-size: 15px !important;
            font-family: 'Inter', 'sans serif', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.03) !important;
        }

        .stTextArea textarea:focus {
            border-color: #4F46E5 !important;
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.08), 0 4px 16px rgba(79, 70, 229, 0.06) !important;
        }

        .stTextArea textarea::placeholder {
            color: #9CA3AF !important;
        }

        /* ==========================================================
           Button
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 56px;
            border: none;
            border-radius: 16px;
            background: linear-gradient(135deg, #4F46E5 0%, #BCA4F5 50%, #81CFFF 100%);
            background-size: 200% 200%;
            color: white;
            font-size: 16px;
            font-weight: 700;
            font-family: 'Inter', 'sans serif', sans-serif;
            transition: all 0.4s ease;
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.2);
            letter-spacing: 0.5px;
        }

        .stButton > button:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 8px 32px rgba(79, 70, 229, 0.3);
            background-position: 100% 100%;
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
            background: #FFFFFF;
            padding: 1.25rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(188, 164, 245, 0.12);
            text-align: center;
            transition: all 0.4s ease;
            cursor: default;
        }

        .metric-item:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 12px 40px rgba(79, 70, 229, 0.08);
            border-color: rgba(188, 164, 245, 0.25);
        }

        .metric-icon { font-size: 1.6rem; margin-bottom: 4px; display: block; }

        .metric-number {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #4F46E5, #BCA4F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
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
            gap: 6px;
            background: rgba(255, 255, 255, 0.5);
            padding: 6px;
            border-radius: 16px;
            border: 1px solid rgba(188, 164, 245, 0.12);
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-size: 0.85rem;
            color: #6B7280;
            font-family: 'Inter', 'sans serif', sans-serif;
            transition: all 0.3s ease;
            background: transparent;
            border: none;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.6);
            color: #4F46E5;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #4F46E5, #BCA4F5) !important;
            color: white !important;
            box-shadow: 0 4px 16px rgba(79, 70, 229, 0.15);
            font-weight: 600;
        }

        /* ==========================================================
           Result Boxes
        ========================================================== */
        .result-box {
            background: #FFFFFF;
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border-left: 4px solid #4F46E5;
            border-top: 1px solid rgba(188, 164, 245, 0.12);
            border-right: 1px solid rgba(188, 164, 245, 0.12);
            border-bottom: 1px solid rgba(188, 164, 245, 0.12);
            margin-bottom: 0.75rem;
            color: #1F2937;
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.3s ease;
        }

        .result-box:hover {
            background: rgba(255, 255, 255, 0.95);
            border-left-color: #BCA4F5;
            transform: translateX(6px);
            box-shadow: 0 4px 16px rgba(79, 70, 229, 0.05);
        }

        /* ==========================================================
           Token Boxes
        ========================================================== */
        .token-box {
            display: inline-block;
            background: #F3F4F6;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 10px;
            border: 1px solid rgba(188, 164, 245, 0.12);
            font-size: 0.9rem;
            color: #1F2937;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .token-box:hover {
            background: rgba(79, 70, 229, 0.06);
            border-color: #4F46E5;
            transform: translateY(-2px);
        }

        /* ==========================================================
           Quick Stats
        ========================================================== */
        .quick-stats {
            background: #FFFFFF;
            padding: 1rem 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(188, 164, 245, 0.12);
        }

        .quick-stats p {
            margin: 0.4rem 0;
            color: #4B5563;
            font-size: 0.95rem;
        }

        .quick-stats strong {
            color: #4F46E5;
            font-weight: 600;
        }

        .common-words {
            background: #FFFFFF;
            padding: 0.75rem 1.25rem;
            border-radius: 14px;
            margin-top: 0.75rem;
            border: 1px solid rgba(188, 164, 245, 0.12);
            color: #1F2937;
            font-size: 0.95rem;
        }

        .common-words strong {
            color: #4F46E5;
        }

        .common-words span {
            background: rgba(79, 70, 229, 0.05);
            padding: 0.2rem 0.8rem;
            border-radius: 12px;
            display: inline-block;
            margin: 0.15rem;
            border: 1px solid rgba(188, 164, 245, 0.08);
        }

        /* ==========================================================
           Footer
        ========================================================== */
        .footer {
            text-align: center;
            color: #9CA3AF;
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid rgba(188, 164, 245, 0.12);
            margin-top: 2rem;
        }

        .footer span {
            background: linear-gradient(135deg, #4F46E5, #BCA4F5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 600;
        }

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
<div style="display:flex;align-items:center;gap:10px;margin:1.5rem 0 1rem 0;">
    <span style="font-size:1.5rem;">✍️</span>
    <span style="font-size:1.3rem;font-weight:700;color:#1F2937;">Input Text</span>
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
    
    with st.spinner("🔄 Processing your text..."):
        time.sleep(0.3)
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        
        pos_tags = nltk.pos_tag(words)
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        chunks = [chunk.text for chunk in doc.noun_chunks]
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        unique_words = len(set(words))
        char_count = len(text)
        word_freq = Counter([word.lower() for word in words if word.isalpha()])
        most_common = word_freq.most_common(5)
    
    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin:0 0 0.75rem 0;">
            <span style="font-size:1.3rem;">📄</span>
            <span style="font-size:1.2rem;font-weight:700;color:#1F2937;">Original Text</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin:0 0 0.75rem 0;">
            <span style="font-size:1.3rem;">📊</span>
            <span style="font-size:1.2rem;font-weight:700;color:#1F2937;">Quick Stats</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="quick-stats">
            <p><strong>Characters:</strong> {char_count:,}</p>
            <p><strong>Words:</strong> {len(words):,}</p>
            <p><strong>Sentences:</strong> {len(sentences):,}</p>
            <p><strong>Avg. Word Length:</strong> {avg_word_length:.1f}</p>
            <p><strong>Unique Words:</strong> {unique_words:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    
    st.markdown("---")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin:0 0 1rem 0;">
        <span style="font-size:1.3rem;">📈</span>
        <span style="font-size:1.2rem;font-weight:700;color:#1F2937;">Analysis Metrics</span>
    </div>
    """, unsafe_allow_html=True)
    
    metrics_data = [
        ("📝", "Total Words", len(words)),
        ("📑", "Sentences", len(sentences)),
        ("🧹", "Stopwords Removed", len(words) - len(filtered_words)),
        ("🏷️", "Named Entities", len(entities)),
        ("🔗", "Noun Phrases", len(chunks)),
        ("✨", "Unique Words", unique_words),
    ]
    
    metrics_html = '<div class="metric-grid">'
    for icon, label, value in metrics_data:
        metrics_html += f"""
        <div class="metric-item">
            <span class="metric-icon">{icon}</span>
            <div class="metric-number">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """
    metrics_html += '</div>'
    
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Most common words
    if most_common:
        st.markdown(f"""
        <div class="common-words">
            <strong>🔥 Most Common Words:</strong> 
            {', '.join([f'<span>{word} ({count})</span>' for word, count in most_common])}
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
    
    with tab1:
        st.subheader("📑 Sentence Segmentation")
        st.caption("Breaking text into individual sentences")
        for i, sent in enumerate(sentences, 1):
            st.markdown(f'<div class="result-box"><strong>{i}.</strong> {sent}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("🔤 Word Tokenization")
        st.caption("Breaking text into individual words/tokens")
        tokens_html = " ".join([f'<span class="token-box">{token}</span>' for token in words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
        st.caption(f"Total tokens: {len(words)}")
    
    with tab3:
        st.subheader("🚫 Stop Word Removal")
        st.caption("Common words removed to focus on meaningful content")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in filtered_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
        st.caption(f"Tokens after stopword removal: {len(filtered_words)} (removed {len(words) - len(filtered_words)} stopwords)")
    
    with tab4:
        st.subheader("🌱 Stemming")
        st.caption("Reducing words to their root form using Porter Stemmer")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in stemmed_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
    
    with tab5:
        st.subheader("📖 Lemmatization")
        st.caption("Reducing words to their dictionary form using WordNet")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in lemmatized_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
    
    with tab6:
        st.subheader("🏷️ Part-of-Speech Tagging")
        st.caption("Grammatical tags assigned to each word")
        pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    
    with tab7:
        st.subheader("👤 Named Entity Recognition")
        st.caption("Identifying named entities in text (people, organizations, locations, etc.)")
        if entities:
            ner_df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.dataframe(ner_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No named entities found in the text.")
    
    with tab8:
        st.subheader("🔗 Dependency Parsing")
        st.caption("Grammatical relationships between words in the sentence")
        dep_df = pd.DataFrame(dependencies, columns=["Word", "Dependency", "Head"])
        st.dataframe(dep_df, use_container_width=True, hide_index=True)
    
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
    <span>NLP Toolkit</span>
</div>
""", unsafe_allow_html=True)