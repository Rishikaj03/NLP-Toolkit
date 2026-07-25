import streamlit as st
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import os
import spacy

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NLP Toolkit - Professional Text Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS FOR PROFESSIONAL UI
# --------------------------------------------------

def load_css():
    st.markdown("""
    <style>
        /* ==========================================================
           Google Font
        ========================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* ==========================================================
           Global
        ========================================================== */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* ==========================================================
           Background with subtle gradient
        ========================================================== */
        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
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
        }
        
        /* ==========================================================
           Custom Header - Refined gradient
        ========================================================== */
        .header-container {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #334155 100%);
            padding: 2.8rem 2rem 2.2rem 2rem;
            border-radius: 24px;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.06);
            position: relative;
            overflow: hidden;
        }
        
        .header-container::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(96, 165, 250, 0.08) 0%, transparent 70%);
            border-radius: 50%;
        }
        
        .header-container::after {
            content: '';
            position: absolute;
            bottom: -30%;
            left: -10%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(167, 139, 250, 0.06) 0%, transparent 70%);
            border-radius: 50%;
        }
        
        .header-title {
            color: #FFFFFF;
            font-size: 2.8rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
        }
        
        .header-title span {
            background: linear-gradient(135deg, #60A5FA, #A78BFA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-subtitle {
            color: rgba(255, 255, 255, 0.7);
            text-align: center;
            font-size: 1.1rem;
            margin-top: 0.5rem;
            font-weight: 400;
            letter-spacing: 0.3px;
            position: relative;
            z-index: 1;
        }
        
        /* ==========================================================
           Card Styling - Glass morphism effect
        ========================================================== */
        .card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            border-color: rgba(255, 255, 255, 0.9);
        }
        
        /* ==========================================================
           Text Area - With subtle styling
        ========================================================== */
        .stTextArea textarea {
            background: #FFFFFF !important;
            color: #1E293B !important;
            border: 2px solid #E2E8F0 !important;
            border-radius: 14px !important;
            padding: 18px !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        }
        
        .stTextArea textarea:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.08), 0 4px 12px rgba(59, 130, 246, 0.05) !important;
        }
        
        .stTextArea textarea::placeholder {
            color: #94A3B8 !important;
            font-weight: 300;
        }
        
        /* ==========================================================
           Button - Refined gradient
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 54px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%);
            color: white;
            font-size: 16px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.25);
            letter-spacing: 0.3px;
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
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
            transition: left 0.5s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(59, 130, 246, 0.35);
            background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:active {
            transform: translateY(0px);
        }
        
        /* ==========================================================
           Metric Cards - With subtle shadows and gradient numbers
        ========================================================== */
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(8px);
            padding: 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3B82F6, #8B5CF6);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .metric-card:hover::before {
            opacity: 1;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            border-color: rgba(255, 255, 255, 0.9);
        }
        
        .metric-number {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.2;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 0.35rem;
            font-weight: 500;
            letter-spacing: 0.3px;
        }
        
        /* ==========================================================
           Tabs - Improved styling
        ========================================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(241, 245, 249, 0.7);
            backdrop-filter: blur(8px);
            padding: 6px;
            border-radius: 16px;
            border: 1px solid #E2E8F0;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-size: 0.9rem;
            color: #64748B;
            font-family: 'Inter', sans-serif;
            transition: all 0.25s ease;
            background: transparent;
            border: none;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.6);
            color: #1E293B;
        }
        
        .stTabs [aria-selected="true"] {
            background: #FFFFFF !important;
            color: #3B82F6 !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            font-weight: 600;
        }
        
        /* ==========================================================
           Result Boxes - With gradient border
        ========================================================== */
        .result-box {
            background: rgba(248, 250, 252, 0.8);
            backdrop-filter: blur(4px);
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border-left: 4px solid transparent;
            background-image: linear-gradient(#F8FAFC, #F8FAFC), linear-gradient(135deg, #3B82F6, #8B5CF6);
            background-origin: padding-box, border-box;
            background-clip: padding-box, border-box;
            margin-bottom: 0.75rem;
            color: #1E293B;
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.2s ease;
        }
        
        .result-box:hover {
            background: rgba(248, 250, 252, 0.95);
            transform: translateX(4px);
        }
        
        .token-box {
            display: inline-block;
            background: rgba(241, 245, 249, 0.8);
            backdrop-filter: blur(4px);
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            font-size: 0.9rem;
            color: #1E293B;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .token-box:hover {
            background: #E2E8F0;
            border-color: #CBD5E1;
            transform: translateY(-1px);
        }
        
        /* ==========================================================
           Quick Stats - Glass card
        ========================================================== */
        .quick-stats {
            background: rgba(248, 250, 252, 0.8);
            backdrop-filter: blur(8px);
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        
        .quick-stats p {
            margin: 0.4rem 0;
            color: #475569;
            font-size: 0.95rem;
        }
        
        .quick-stats strong {
            color: #1E293B;
            font-weight: 600;
        }
        
        /* ==========================================================
           Common Words Box
        ========================================================== */
        .common-words {
            background: rgba(248, 250, 252, 0.8);
            backdrop-filter: blur(4px);
            padding: 0.75rem 1.25rem;
            border-radius: 12px;
            margin-top: 0.75rem;
            border: 1px solid #E2E8F0;
            color: #1E293B;
            font-size: 0.95rem;
        }
        
        .common-words strong {
            color: #3B82F6;
        }
        
        /* ==========================================================
           Footer
        ========================================================== */
        .footer {
            text-align: center;
            color: #94A3B8;
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid #E2E8F0;
            margin-top: 2rem;
        }
        
        .footer span {
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 600;
        }
        
        /* ==========================================================
           Scrollbar
        ========================================================== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F1F5F9;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94A3B8;
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