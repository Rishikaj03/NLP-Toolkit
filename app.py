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
    page_title="NLP Studio",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS — Aurora Dark Studio
# --------------------------------------------------
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-0: #07070C;
        --bg-1: #0E0E1A;
        --bg-2: #14142380;
        --stroke: rgba(255,255,255,0.08);
        --stroke-strong: rgba(255,255,255,0.16);
        --text: #EAEAF2;
        --muted: #8B8BA7;
        --aurora-1: #7C5CFF;
        --aurora-2: #22D3EE;
        --aurora-3: #F472B6;
        --aurora-4: #34D399;
    }

    html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        color: var(--text);
    }

    /* Aurora background */
    .stApp {
        background:
            radial-gradient(1000px 600px at 10% -10%, rgba(124,92,255,0.25), transparent 60%),
            radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.18), transparent 60%),
            radial-gradient(800px 500px at 50% 100%, rgba(244,114,182,0.15), transparent 60%),
            var(--bg-0);
        background-attachment: fixed;
    }

    header, footer { visibility: hidden; }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        padding: 3rem 2.5rem;
        border-radius: 28px;
        background: linear-gradient(180deg, rgba(20,20,35,0.85), rgba(10,10,20,0.7));
        border: 1px solid var(--stroke);
        overflow: hidden;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
    }
    .hero::before {
        content:'';
        position:absolute; inset:-2px;
        background: conic-gradient(from 180deg at 50% 50%, #7C5CFF, #22D3EE, #F472B6, #7C5CFF);
        filter: blur(60px); opacity: 0.35; z-index:0;
        animation: spin 18s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .hero-inner { position: relative; z-index: 1; }

    .hero-badge {
        display:inline-flex; align-items:center; gap:8px;
        padding: 6px 14px; border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid var(--stroke-strong);
        font-size: 0.8rem; color: var(--muted);
        letter-spacing: 0.15em; text-transform: uppercase;
        margin-bottom: 1.25rem;
    }
    .hero-badge .dot {
        width:6px; height:6px; border-radius:50%;
        background: var(--aurora-4);
        box-shadow: 0 0 12px var(--aurora-4);
    }
    .hero-title {
        font-size: 3.4rem; font-weight: 700; line-height: 1.05;
        letter-spacing: -0.03em; margin: 0;
    }
    .hero-title em {
        font-style: normal;
        background: linear-gradient(120deg, var(--aurora-2), var(--aurora-1) 50%, var(--aurora-3));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        margin-top: 1rem; color: var(--muted);
        font-size: 1.1rem; max-width: 620px;
    }

    /* ---------- Text area ---------- */
    .stTextArea label { color: var(--text) !important; font-weight: 500; }
    .stTextArea textarea {
        background: rgba(10,10,20,0.6) !important;
        color: var(--text) !important;
        border: 1px solid var(--stroke) !important;
        border-radius: 18px !important;
        padding: 20px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 14px !important;
        line-height: 1.7 !important;
        backdrop-filter: blur(10px);
        transition: all .25s ease;
    }
    .stTextArea textarea:focus {
        border-color: var(--aurora-1) !important;
        box-shadow: 0 0 0 4px rgba(124,92,255,0.18) !important;
    }
    .stTextArea textarea::placeholder { color: #4A4A66 !important; }

    /* ---------- Button ---------- */
    .stButton > button {
        width: 100%; height: 56px;
        border: 1px solid var(--stroke-strong);
        border-radius: 16px;
        background: linear-gradient(135deg, var(--aurora-1), var(--aurora-2));
        color: #0B0B14; font-weight: 700; font-size: 15px;
        letter-spacing: 0.02em;
        transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
        box-shadow: 0 12px 32px rgba(124,92,255,0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.05);
        box-shadow: 0 18px 40px rgba(124,92,255,0.5);
    }

    /* ---------- Metric cards ---------- */
    .metric {
        position: relative;
        padding: 20px 18px;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid var(--stroke);
        overflow: hidden;
        transition: transform .25s ease, border-color .25s ease;
        margin-bottom: 14px;
    }
    .metric:hover { transform: translateY(-3px); border-color: var(--stroke-strong); }
    .metric::after {
        content:''; position:absolute; inset:auto -30% -60% auto;
        width:180px; height:180px; border-radius:50%;
        background: radial-gradient(circle, var(--glow, var(--aurora-1)) 0%, transparent 70%);
        opacity:0.25;
    }
    .metric .num {
        font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em;
        color: var(--text); margin: 0; line-height: 1.1;
    }
    .metric .lbl {
        font-size: 0.75rem; color: var(--muted);
        margin-top: 6px; text-transform: uppercase; letter-spacing: 0.15em;
    }

    /* ---------- Panels ---------- */
    .panel {
        background: rgba(14,14,26,0.6);
        border: 1px solid var(--stroke);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(14px);
    }
    .panel h3, .panel-title {
        margin: 0 0 1rem 0; font-weight: 600; font-size: 1rem;
        color: var(--muted); text-transform: uppercase; letter-spacing: 0.18em;
    }
    .original-text {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text); line-height: 1.75; font-size: 0.95rem;
        white-space: pre-wrap;
    }
    .stat-row {
        display:flex; justify-content:space-between;
        padding: 10px 0; border-bottom: 1px dashed var(--stroke);
        font-size: 0.9rem;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-row span:first-child { color: var(--muted); }
    .stat-row span:last-child { color: var(--text); font-weight: 600; }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(10,10,20,0.5);
        border: 1px solid var(--stroke);
        padding: 6px; border-radius: 16px;
        backdrop-filter: blur(10px);
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px; padding: 0.55rem 1rem;
        color: var(--muted); font-weight: 500; font-size: 0.85rem;
        background: transparent; border: none;
        transition: all .2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text); background: rgba(255,255,255,0.04); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(124,92,255,0.25), rgba(34,211,238,0.2)) !important;
        color: var(--text) !important;
        border: 1px solid var(--stroke-strong);
    }

    /* ---------- Chips / boxes ---------- */
    .chip {
        display:inline-block; margin: 4px 6px 4px 0;
        padding: 6px 12px; border-radius: 10px;
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--stroke);
        color: var(--text); font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        transition: all .2s ease;
    }
    .chip:hover {
        border-color: var(--aurora-1);
        background: rgba(124,92,255,0.12);
        transform: translateY(-1px);
    }
    .sent-line {
        display:flex; gap: 14px; align-items:flex-start;
        padding: 14px 16px; margin-bottom: 10px;
        background: rgba(255,255,255,0.03);
        border-left: 2px solid var(--aurora-1);
        border-radius: 12px;
        transition: all .2s ease;
    }
    .sent-line:hover { background: rgba(124,92,255,0.08); transform: translateX(4px); }
    .sent-num {
        color: var(--aurora-2); font-weight: 600;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
        min-width: 24px;
    }
    .sent-text { color: var(--text); line-height: 1.6; }

    .chunk-pill {
        display:inline-block; margin: 6px 6px 0 0;
        padding: 8px 14px; border-radius: 999px;
        background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(34,211,238,0.1));
        border: 1px solid rgba(52,211,153,0.3);
        color: #A7F3D0; font-size: 0.9rem;
    }

    .common-strip {
        display:flex; flex-wrap:wrap; gap: 8px; align-items:center;
        padding: 14px 18px; margin-top: 1rem;
        background: rgba(14,14,26,0.6);
        border: 1px solid var(--stroke);
        border-radius: 14px;
    }
    .common-strip .k { color: var(--muted); font-size:0.8rem; text-transform:uppercase; letter-spacing:0.15em; margin-right: 8px; }
    .common-strip .w {
        padding: 4px 10px; border-radius: 8px;
        background: rgba(124,92,255,0.15);
        border: 1px solid rgba(124,92,255,0.35);
        color: #C4B5FD; font-family: 'JetBrains Mono', monospace; font-size:0.85rem;
    }
    .common-strip .w b { color: #EAEAF2; margin-left: 4px; }

    /* ---------- Data frames ---------- */
    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 14px; overflow: hidden;
        border: 1px solid var(--stroke);
    }

    /* ---------- Divider ---------- */
    hr {
        border: none;
        border-top: 1px solid var(--stroke);
        margin: 2rem 0;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center; color: var(--muted);
        padding: 2rem 0 0.5rem; font-size: 0.85rem;
        border-top: 1px solid var(--stroke); margin-top: 3rem;
    }
    .footer b {
        background: linear-gradient(120deg, var(--aurora-2), var(--aurora-3));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width:8px; height:8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius:10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    @media (max-width: 768px) {
        .hero { padding: 2rem 1.25rem; }
        .hero-title { font-size: 2.2rem; }
        .metric .num { font-size: 1.6rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.7rem; font-size: 0.75rem; }
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# CACHE RESOURCES
# --------------------------------------------------
@st.cache_resource
def download_nltk():
    for p in ["punkt","punkt_tab","stopwords","wordnet","omw-1.4",
              "averaged_perceptron_tagger","averaged_perceptron_tagger_eng"]:
        nltk.download(p, quiet=True)
download_nltk()

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")
nlp = load_spacy()

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-badge"><span class="dot"></span> NLP · Studio · v2</div>
    <h1 class="hero-title">Decode language.<br><em>Reveal meaning.</em></h1>
    <p class="hero-sub">A sleek toolkit for tokenization, POS tagging, named-entity recognition, and dependency parsing — powered by NLTK & spaCy.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT
# --------------------------------------------------
text = st.text_area(
    "Enter text to analyze",
    height=200,
    placeholder="Paste or type English text here…\n\ne.g. Apple Inc. is planning to open a new store in New York next month."
)

c1, c2, c3 = st.columns([1,2,1])
with c2:
    analyze = st.button("Analyze Text  →", use_container_width=True)

# --------------------------------------------------
# ANALYSIS (logic unchanged)
# --------------------------------------------------
if analyze:
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
        st.stop()

    with st.spinner("Analyzing…"):
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if w.lower() not in stop_words]
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(w) for w in filtered_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
        pos_tags = nltk.pos_tag(words)
        doc = nlp(text)
        entities = [(e.text, e.label_) for e in doc.ents]
        dependencies = [(t.text, t.dep_, t.head.text) for t in doc]
        chunks = [c.text for c in doc.noun_chunks]
        avg_word_length = sum(len(w) for w in words)/len(words) if words else 0
        unique_words = len(set(words))
        char_count = len(text)
        word_freq = Counter([w.lower() for w in words if w.isalpha()])
        most_common = word_freq.most_common(5)

    st.markdown("---")

    # Original + quick stats
    a, b = st.columns([2,1])
    with a:
        st.markdown('<div class="panel"><div class="panel-title">Original Text</div>'
                    f'<div class="original-text">{text}</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">Quick Stats</div>
          <div class="stat-row"><span>Characters</span><span>{char_count:,}</span></div>
          <div class="stat-row"><span>Words</span><span>{len(words):,}</span></div>
          <div class="stat-row"><span>Sentences</span><span>{len(sentences):,}</span></div>
          <div class="stat-row"><span>Avg. word length</span><span>{avg_word_length:.1f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(6)
    metrics = [
        (len(words), "Total Words", "#7C5CFF"),
        (len(sentences), "Sentences", "#22D3EE"),
        (len(filtered_words), "After Stopwords", "#F472B6"),
        (len(entities), "Named Entities", "#34D399"),
        (len(chunks), "Noun Phrases", "#FBBF24"),
        (unique_words, "Unique Words", "#60A5FA"),
    ]
    for col, (num, lbl, glow) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric" style="--glow:{glow}"><p class="num">{num}</p><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    if most_common:
        words_html = "".join([f'<span class="w">{w}<b>×{c}</b></span>' for w, c in most_common])
        st.markdown(f'<div class="common-strip"><span class="k">Most Common</span>{words_html}</div>', unsafe_allow_html=True)

    st.markdown("---")

    tabs = st.tabs([
        "Sentences","Tokens","Stop Words","Stemming","Lemmatization",
        "POS Tags","NER","Dependency","Chunking"
    ])

    with tabs[0]:
        st.markdown('<div class="panel-title">Sentence Segmentation</div>', unsafe_allow_html=True)
        for i, s in enumerate(sentences, 1):
            st.markdown(f'<div class="sent-line"><div class="sent-num">{i:02d}</div><div class="sent-text">{s}</div></div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="panel-title">Word Tokenization</div>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="chip">{t}</span>' for t in words]), unsafe_allow_html=True)
        st.caption(f"Total tokens: {len(words)}")

    with tabs[2]:
        st.markdown('<div class="panel-title">Stop Word Removal</div>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="chip">{w}</span>' for w in filtered_words]), unsafe_allow_html=True)
        st.caption(f"Remaining: {len(filtered_words)} · Removed: {len(words)-len(filtered_words)}")

    with tabs[3]:
        st.markdown('<div class="panel-title">Stemming — Porter</div>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="chip">{w}</span>' for w in stemmed_words]), unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="panel-title">Lemmatization — WordNet</div>', unsafe_allow_html=True)
        st.markdown("".join([f'<span class="chip">{w}</span>' for w in lemmatized_words]), unsafe_allow_html=True)

    with tabs[5]:
        st.markdown('<div class="panel-title">Part-of-Speech Tags</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(pos_tags, columns=["Word","POS Tag"]), use_container_width=True, hide_index=True)

    with tabs[6]:
        st.markdown('<div class="panel-title">Named Entity Recognition</div>', unsafe_allow_html=True)
        if entities:
            st.dataframe(pd.DataFrame(entities, columns=["Entity","Label"]), use_container_width=True, hide_index=True)
        else:
            st.info("No named entities found.")

    with tabs[7]:
        st.markdown('<div class="panel-title">Dependency Parsing</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(dependencies, columns=["Word","Dependency","Head"]), use_container_width=True, hide_index=True)

    with tabs[8]:
        st.markdown('<div class="panel-title">Noun Phrase Chunking</div>', unsafe_allow_html=True)
        if chunks:
            st.markdown("".join([f'<span class="chunk-pill">{c}</span>' for c in chunks]), unsafe_allow_html=True)
            st.caption(f"Total noun phrases: {len(chunks)}")
        else:
            st.info("No noun phrases found.")

    # Download (logic unchanged)
    st.markdown("---")
    result = f"""====================================
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
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(sentences)])}

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
{chr(10).join([f"{w}: {t}" for w, t in pos_tags])}

------------------------------------
Named Entity Recognition:
------------------------------------
{chr(10).join([f"{e}: {l}" for e, l in entities]) if entities else "No entities found"}

------------------------------------
Dependency Parsing:
------------------------------------
{chr(10).join([f"{w} -> {d} -> {h}" for w, d, h in dependencies])}

------------------------------------
Noun Phrase Chunking:
------------------------------------
{chr(10).join(chunks) if chunks else "No noun phrases found"}

====================================
"""
    d1, d2, d3 = st.columns([1,2,1])
    with d2:
        st.download_button("⬇  Download Full Report", data=result,
                           file_name="nlp_analysis_report.txt", mime="text/plain",
                           use_container_width=True)
    st.success("Analysis complete.")

st.markdown('<div class="footer">Crafted with <b>Streamlit · NLTK · spaCy</b> — NLP Studio</div>', unsafe_allow_html=True)
