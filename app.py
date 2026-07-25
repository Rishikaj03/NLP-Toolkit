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
    page_title="NLP Toolkit - Text Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS - CLEAN MODERN LAYOUT
# --------------------------------------------------

def load_css():
    st.markdown("""
    <style>
        /* ==========================================================
           IMPORTS & RESET
        ========================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
            font-size: 15px;
            line-height: 1.6;
        }
        
        /* ==========================================================
           BACKGROUND
        ========================================================== */
        .stApp {
            background: #F8FAFC;
        }
        
        header { visibility: hidden; }
        footer { visibility: hidden; }
        
        .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        
        /* ==========================================================
           HEADER / NAVIGATION
        ========================================================== */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.5rem;
            background: white;
            border-radius: 16px;
            border: 1px solid #E5E7EB;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
        
        .app-logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .app-logo-icon {
            font-size: 2rem;
        }
        
        .app-logo-text {
            font-weight: 800;
            font-size: 1.3rem;
            color: #1F2937;
            letter-spacing: -0.3px;
        }
        
        .app-logo-text span {
            color: #6366F1;
        }
        
        .app-tagline {
            font-size: 0.85rem;
            color: #6B7280;
            font-weight: 400;
        }
        
        /* ==========================================================
           HERO / INPUT SECTION
        ========================================================== */
        .hero-section {
            background: white;
            padding: 2rem 2.5rem;
            border-radius: 20px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
            margin-bottom: 2rem;
        }
        
        .hero-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1F2937;
            margin: 0 0 0.5rem 0;
        }
        
        .hero-description {
            color: #6B7280;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        
        /* ==========================================================
           TEXT AREA
        ========================================================== */
        .stTextArea textarea {
            background: #F9FAFB !important;
            color: #1F2937 !important;
            border: 2px solid #E5E7EB !important;
            border-radius: 14px !important;
            padding: 16px 18px !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.25s ease !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #6366F1 !important;
            background: white !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.06) !important;
        }
        
        .stTextArea textarea::placeholder {
            color: #9CA3AF !important;
        }
        
        /* ==========================================================
           BUTTONS
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 54px;
            border: none;
            border-radius: 14px;
            background: #6366F1;
            color: white;
            font-size: 16px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);
            background: #4F46E5;
        }
        
        .stButton > button:active {
            transform: translateY(0px);
        }
        
        /* ==========================================================
           QUICK STATS GRID
        ========================================================== */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
            margin: 1rem 0 1.5rem 0;
        }
        
        .stat-card {
            background: white;
            padding: 1.2rem 1rem;
            border-radius: 14px;
            border: 1px solid #E5E7EB;
            text-align: center;
            transition: all 0.2s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            border-color: #C7D2FE;
        }
        
        .stat-number {
            font-size: 1.8rem;
            font-weight: 800;
            color: #1F2937;
            line-height: 1.2;
        }
        
        .stat-label {
            font-size: 0.75rem;
            color: #6B7280;
            font-weight: 500;
            margin-top: 4px;
            letter-spacing: 0.3px;
        }
        
        /* ==========================================================
           RESULT BOXES
        ========================================================== */
        .result-box {
            background: #F9FAFB;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border-left: 4px solid #6366F1;
            border-top: 1px solid #E5E7EB;
            border-right: 1px solid #E5E7EB;
            border-bottom: 1px solid #E5E7EB;
            margin-bottom: 0.75rem;
            color: #1F2937;
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.2s ease;
        }
        
        .result-box:hover {
            background: white;
            border-left-color: #8B5CF6;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.04);
        }
        
        /* ==========================================================
           TOKEN BOXES
        ========================================================== */
        .token-box {
            display: inline-block;
            background: #F3F4F6;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            font-size: 0.85rem;
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
           COMMON WORDS
        ========================================================== */
        .common-words {
            background: white;
            padding: 0.75rem 1.25rem;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            color: #1F2937;
            font-size: 0.95rem;
            margin: 0.5rem 0 1.5rem 0;
        }
        
        .common-words strong {
            color: #6366F1;
        }
        
        /* ==========================================================
           TABS
        ========================================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #F9FAFB;
            padding: 6px;
            border-radius: 14px;
            border: 1px solid #E5E7EB;
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
           QUICK STATS SIDEBAR (for original text)
        ========================================================== */
        .quick-stats {
            background: #F9FAFB;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
        }
        
        .quick-stats p {
            margin: 0.4rem 0;
            color: #4B5563;
            font-size: 0.9rem;
        }
        
        .quick-stats strong {
            color: #1F2937;
        }
        
        /* ==========================================================
           SECTION HEADERS
        ========================================================== */
        .section-header {
            font-size: 1.2rem;
            font-weight: 700;
            color: #1F2937;
            margin: 1.5rem 0 0.75rem 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* ==========================================================
           DATAFRAME
        ========================================================== */
        [data-testid="stDataFrame"] {
            border-radius: 12px !important;
            overflow: hidden !important;
            border: 1px solid #E5E7EB !important;
        }
        
        /* ==========================================================
           DOWNLOAD BUTTON
        ========================================================== */
        [data-testid="stDownloadButton"] > button {
            background: #1F2937 !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 16px rgba(31, 41, 55, 0.15) !important;
            height: 50px !important;
            font-size: 15px !important;
            color: white !important;
            border: none !important;
        }
        
        [data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(31, 41, 55, 0.2) !important;
            background: #111827 !important;
        }
        
        /* ==========================================================
           FOOTER
        ========================================================== */
        .footer {
            text-align: center;
            color: #9CA3AF;
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid #E5E7EB;
            margin-top: 2rem;
        }
        
        .footer span {
            color: #6366F1;
            font-weight: 600;
        }
        
        /* ==========================================================
           RESPONSIVE
        ========================================================== */
        @media (max-width: 768px) {
            .app-header {
                flex-direction: column;
                text-align: center;
                gap: 8px;
                padding: 1rem;
            }
            
            .hero-section {
                padding: 1.5rem;
            }
            
            .stats-grid {
                grid-template-columns: repeat(3, 1fr);
                gap: 8px;
            }
            
            .stat-number {
                font-size: 1.4rem;
            }
            
            .block-container {
                padding: 0.75rem !important;
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
<div class="app-header">
    <div class="app-logo">
        <span class="app-logo-icon">🧠</span>
        <span class="app-logo-text">NLP <span>Toolkit</span></span>
    </div>
    <div class="app-tagline">Advanced Text Analysis • NLTK • spaCy</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO INPUT SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">✨ Natural Language Processing</h1>
    <p class="hero-description">Paste your text below and let AI analyze it for sentences, tokens, entities, and more.</p>
</div>
""", unsafe_allow_html=True)

# Input Area
text = st.text_area(
    "Enter your text",
    height=200,
    placeholder="Paste your English text here...\n\nExample: Apple Inc. is planning to open a new store in New York next month. The company's CEO, Tim Cook, announced this exciting news yesterday.",
    label_visibility="collapsed"
)

# Center the analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("🚀 Analyze Text", use_container_width=True)

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
        st.stop()
    
    with st.spinner("🔍 Analyzing your text..."):
        # All NLP processing
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
    # RESULTS SECTION
    # --------------------------------------------------
    
    st.markdown("---")
    
    # Row: Original Text + Quick Stats
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">📄 Original Text</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">📊 Quick Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="quick-stats">
            <p><strong>Characters:</strong> {char_count:,}</p>
            <p><strong>Words:</strong> {len(words):,}</p>
            <p><strong>Sentences:</strong> {len(sentences):,}</p>
            <p><strong>Avg. Word Length:</strong> {avg_word_length:.1f}</p>
            <p><strong>Unique Words:</strong> {unique_words:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Row: Metrics Grid
    st.markdown('<div class="section-header">📈 Analysis Metrics</div>', unsafe_allow_html=True)
    
    metrics = [
        ("📝", "Total Words", len(words)),
        ("📑", "Sentences", len(sentences)),
        ("🧹", "Stopwords Removed", len(words) - len(filtered_words)),
        ("🏷️", "Named Entities", len(entities)),
        ("🔗", "Noun Phrases", len(chunks)),
        ("✨", "Unique Words", unique_words),
    ]
    
    stats_html = '<div class="stats-grid">'
    for icon, label, value in metrics:
        stats_html += f"""
        <div class="stat-card">
            <div style="font-size:1.4rem; margin-bottom:2px;">{icon}</div>
            <div class="stat-number">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """
    stats_html += '</div>'
    st.markdown(stats_html, unsafe_allow_html=True)
    
    # Row: Most Common Words
    if most_common:
        st.markdown(f"""
        <div class="common-words">
            <strong>🔥 Most Common Words:</strong> 
            {', '.join([f'<span style="background:#F3F4F6;padding:0.2rem 0.8rem;border-radius:12px;margin:0.2rem;">{word} ({count})</span>' for word, count in most_common])}
        </div>
        """, unsafe_allow_html=True)
    
    # Row: Detailed Tabs
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
        st.markdown("#### Sentence Segmentation")
        for i, sent in enumerate(sentences, 1):
            st.markdown(f'<div class="result-box"><strong>{i}.</strong> {sent}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### Word Tokenization")
        tokens_html = " ".join([f'<span class="token-box">{token}</span>' for token in words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
        st.caption(f"Total tokens: {len(words)}")
    
    with tab3:
        st.markdown("#### Stop Word Removal")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in filtered_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
        st.caption(f"Tokens after stopword removal: {len(filtered_words)} (removed {len(words) - len(filtered_words)} stopwords)")
    
    with tab4:
        st.markdown("#### Stemming (Porter Stemmer)")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in stemmed_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown("#### Lemmatization (WordNet)")
        tokens_html = " ".join([f'<span class="token-box">{word}</span>' for word in lemmatized_words])
        st.markdown(f'<div style="padding: 0.5rem 0;">{tokens_html}</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown("#### Part-of-Speech Tagging")
        pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    
    with tab7:
        st.markdown("#### Named Entity Recognition")
        if entities:
            ner_df = pd.DataFrame(entities, columns=["Entity", "Label"])
            st.dataframe(ner_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No named entities found in the text.")
    
    with tab8:
        st.markdown("#### Dependency Parsing")
        dep_df = pd.DataFrame(dependencies, columns=["Word", "Dependency", "Head"])
        st.dataframe(dep_df, use_container_width=True, hide_index=True)
    
    with tab9:
        st.markdown("#### Noun Phrase Chunking")
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
    
    # Generate report
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
    
    st.success("✅ Analysis completed successfully!")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, NLTK & spaCy • 
    <span>NLP Toolkit</span>
</div>
""", unsafe_allow_html=True)