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
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS FOR PROFESSIONAL UI
# --------------------------------------------------
def load_css():
    st.markdown("""
    <style>
        /* ==========================================================
           Google Fonts — Baloo 2 (bubble display), Fredoka (playful
           labels), Inter (body / matches config.toml "sans serif")
        ========================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Baloo+2:wght@600;700;800&family=Fredoka:wght@500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 15.5px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }

        :root {
            --primary: #4F46E5;
            --pink: #FF6EC7;
            --cyan: #22D3EE;
            --purple: #A78BFA;
            --lime: #A3E635;
            --amber: #FBBF24;
            --bg: #F4F7FC;
            --surface: #FFFFFF;
            --text: #1F2937;
            --muted: #64748B;
            --line: #E7E9F5;
        }

        /* ==========================================================
           Background — light per config.toml, pastel Y2K glow blobs
        ========================================================== */
        .stApp {
            background:
                radial-gradient(900px 480px at 8% -8%, rgba(255,110,199,0.10), transparent 60%),
                radial-gradient(900px 480px at 98% 6%, rgba(34,211,238,0.10), transparent 60%),
                radial-gradient(800px 500px at 50% 105%, rgba(167,139,250,0.08), transparent 60%),
                var(--bg);
            background-attachment: fixed;
        }

        header { visibility: hidden; }
        footer { visibility: hidden; }

        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            position: relative;
            z-index: 1;
        }

        h1, h2, h3 {
            color: var(--text) !important;
            font-family: 'Fredoka', 'Inter', sans-serif !important;
        }

        /* ==========================================================
           SIDEBAR — "Neural Lens" project panel
        ========================================================== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF 0%, #FBF7FF 100%);
            border-right: 1px solid var(--line);
        }

        .sb-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0.4rem 0 1.4rem 0;
        }
        .sb-brand-emoji {
            font-size: 1.8rem;
            filter: drop-shadow(0 2px 3px rgba(0,0,0,0.08));
        }
        .sb-brand-name {
            font-family: 'Baloo 2', sans-serif;
            font-weight: 800;
            font-size: 1.5rem;
            background: linear-gradient(135deg, var(--primary), var(--pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .sb-heading {
            font-family: 'Fredoka', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text);
            margin: 1.2rem 0 0.6rem 0;
            padding-bottom: 6px;
            border-bottom: 2px dashed #E9D5FF;
        }

        .sb-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.7rem 1rem;
            box-shadow: 0 2px 8px rgba(79,70,229,0.05);
        }
        .sb-card p {
            margin: 0.45rem 0;
            font-size: 0.85rem;
            color: var(--muted);
            line-height: 1.35;
        }
        .sb-card strong {
            color: var(--text);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        .sb-chips { display: flex; flex-wrap: wrap; gap: 8px; }
        .sb-chip {
            font-family: 'Fredoka', sans-serif;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            color: #1E293B;
            border: 1px solid rgba(0,0,0,0.04);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
            transition: transform 0.2s ease;
        }
        .sb-chip:hover { transform: translateY(-2px) scale(1.04); }
        .sb-chip-pink   { background: linear-gradient(135deg, #FFD6EF, #FFC1E3); }
        .sb-chip-cyan   { background: linear-gradient(135deg, #CFFAFE, #A5F3FC); }
        .sb-chip-purple { background: linear-gradient(135deg, #EDE9FE, #DDD6FE); }
        .sb-chip-lime   { background: linear-gradient(135deg, #ECFCCB, #D9F99D); }

        .sb-footnote {
            margin-top: 1.6rem;
            font-size: 0.8rem;
            color: var(--muted);
            line-height: 1.5;
            padding-top: 1rem;
            border-top: 1px solid var(--line);
        }

        /* ==========================================================
           INFO BADGE ROW (Developer / Date / Roll No)
        ========================================================== */
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin: 1.1rem 0 1.6rem 0;
        }
        .info-badge {
            font-family: 'Fredoka', sans-serif;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text);
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.5rem 1.1rem;
            box-shadow: 0 2px 6px rgba(79,70,229,0.06), inset 0 1px 0 rgba(255,255,255,0.8);
            transition: all 0.2s ease;
        }
        .info-badge:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(79,70,229,0.12);
            border-color: #C7D2FE;
        }

        .section-label {
            font-family: 'Fredoka', sans-serif;
            font-weight: 600;
            font-size: 1.3rem;
            color: var(--text);
            margin: 0.4rem 0 0.6rem 2px;
        }
        .section-label::before { content: "📝 "; }

        /* ==========================================================
           Generic Card
        ========================================================== */
        .card {
            background: var(--surface);
            padding: 1.5rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            box-shadow: 0 2px 10px rgba(79,70,229,0.05);
            margin-bottom: 1.5rem;
            transition: all 0.25s ease;
        }
        .card:hover {
            box-shadow: 0 12px 28px rgba(236,110,199,0.12);
            border-color: #F5D0E8;
            transform: translateY(-2px);
        }

        /* ==========================================================
           Text Area
        ========================================================== */
        .stTextArea textarea {
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 2px solid var(--line) !important;
            border-radius: 18px !important;
            padding: 18px !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 2px 8px rgba(79,70,229,0.04);
        }
        .stTextArea textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 4px rgba(79,70,229,0.12), 0 4px 16px rgba(236,110,199,0.10) !important;
        }
        .stTextArea textarea::placeholder {
            color: #A1A8C3 !important;
            font-weight: 300;
        }

        /* ==========================================================
           Button — glossy Y2K bubble button
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 56px;
            border: none;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--pink) 55%, var(--purple) 100%);
            background-size: 200% 200%;
            color: white;
            font-family: 'Fredoka', 'Inter', sans-serif;
            font-size: 17px;
            font-weight: 600;
            letter-spacing: 0.3px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 6px 18px rgba(79,70,229,0.28), inset 0 1px 0 rgba(255,255,255,0.35);
            position: relative;
            overflow: hidden;
        }
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent);
            transition: left 0.6s ease;
        }
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 12px 26px rgba(236,110,199,0.35), inset 0 1px 0 rgba(255,255,255,0.4);
            background-position: 100% 100%;
        }
        .stButton > button:hover::before { left: 100%; }
        .stButton > button:active { transform: translateY(-1px) scale(0.99); }

        /* ==========================================================
           Metric cards, tabs, result boxes, tokens, misc
        ========================================================== */
        .metric-card {
            background: linear-gradient(180deg, #FFFFFF, #FDFBFF);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px 12px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(79,70,229,0.05);
        }
        .metric-icon { font-size: 1.3rem; margin-bottom: 2px; }
        .metric-number {
            font-family: 'Baloo 2', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--pink), var(--purple), var(--cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
        }
        .metric-label {
            font-size: 0.8rem;
            color: var(--muted);
            margin-top: 0.2rem;
            font-weight: 500;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #F1EEFC;
            padding: 6px;
            border-radius: 16px;
            border: 1px solid var(--line);
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-size: 0.88rem;
            color: var(--muted);
            font-family: 'Fredoka', sans-serif;
            transition: all 0.22s ease;
            background: transparent;
            border: none;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.7);
            color: var(--text);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary), var(--pink)) !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 14px rgba(236,110,199,0.28);
            font-weight: 600;
        }

        .result-box {
            background: var(--surface);
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border-left: 4px solid var(--pink);
            border-top: 1px solid var(--line);
            border-right: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.75rem;
            color: var(--text);
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.2s ease;
        }
        .result-box:hover {
            border-left-color: var(--purple);
            transform: translateX(4px);
            box-shadow: 0 4px 14px rgba(167,139,250,0.14);
        }

        .token-box {
            display: inline-block;
            background: #F1EEFC;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            font-size: 0.9rem;
            color: var(--text);
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .token-box:hover {
            background: linear-gradient(135deg, var(--primary), var(--pink));
            border-color: transparent;
            color: #FFFFFF;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(236,110,199,0.3);
        }

        .quick-stats {
            background: var(--surface);
            padding: 1rem 1.25rem;
            border-radius: 16px;
            border: 1px solid var(--line);
            box-shadow: 0 2px 8px rgba(79,70,229,0.05);
        }
        .quick-stats p { margin: 0.4rem 0; color: #475569; font-size: 0.95rem; }
        .quick-stats strong { color: var(--primary); font-weight: 600; }

        .common-words {
            background: var(--surface);
            padding: 0.75rem 1.25rem;
            border-radius: 14px;
            margin-top: 0.75rem;
            border: 1px solid var(--line);
            color: var(--text);
            font-size: 0.95rem;
        }
        .common-words strong { color: var(--pink); }

        [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--line);
        }

        .footer {
            text-align: center;
            color: #94A3B8;
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid var(--line);
            margin-top: 2rem;
        }
        .footer span {
            font-family: 'Fredoka', sans-serif;
            background: linear-gradient(135deg, var(--primary), var(--pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 600;
        }

        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #F1EEFC; border-radius: 10px; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, var(--pink), var(--purple));
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, var(--cyan) 0%, var(--primary) 100%) !important;
            border-radius: 999px !important;
            font-family: 'Fredoka', sans-serif !important;
            box-shadow: 0 6px 16px rgba(34,211,238,0.25) !important;
        }

        [data-testid="stAlert"] { border-radius: 14px !important; }

        /* ==========================================================
           Responsive
        ========================================================== */
        @media (max-width: 768px) {
            h1, h2, h3 { color: var(--text); }
            .block-container { padding: 1rem !important; }
            .stTextArea textarea { min-height: 180px !important; font-size: 16px !important; }
            .stButton button { width: 100%; font-size: 18px; height: 55px; }
            .metric-number { font-size: 1.7rem; }
            .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
            .badge-row { gap: 6px; }
            .info-badge { font-size: 0.75rem; padding: 0.4rem 0.85rem; }
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
# PROJECT META (edit these three lines for your submission)
# --------------------------------------------------

DEVELOPER_NAME = "Your Name"
ROLL_NO = "00"
PROJECT_DATE = "26/07/2026"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-emoji">💽</span>
        <span class="sb-brand-name">Neural Lens</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-heading">Project Information</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-card">
        <p><strong>Developer</strong><br>{DEVELOPER_NAME}</p>
        <p><strong>Roll No.</strong><br>{ROLL_NO}</p>
        <p><strong>Date</strong><br>{PROJECT_DATE}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-heading">Technology</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-chips">
        <span class="sb-chip sb-chip-pink">Python</span>
        <span class="sb-chip sb-chip-cyan">NLTK</span>
        <span class="sb-chip sb-chip-purple">spaCy</span>
        <span class="sb-chip sb-chip-lime">Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-footnote">A retro-futuristic workspace for practical natural language processing. ✨</div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

HERO_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Baloo+2:wght@700;800&family=Fredoka:wght@500;600&display=swap" rel="stylesheet">
<div id="hero" style="position:relative;width:100%;height:230px;border-radius:26px;overflow:hidden;
     background:linear-gradient(135deg,#FFE1F3 0%,#E5E9FF 45%,#DFF9F6 100%);
     border:2px solid #FFFFFF;box-shadow:0 2px 4px rgba(79,70,229,0.06),0 18px 36px rgba(236,72,201,0.14);
     font-family:'Baloo 2','Inter',sans-serif;">
  <canvas id="particles" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>

  <div style="position:absolute;top:0;left:0;right:0;height:6px;
       background:linear-gradient(90deg,#FF6EC7,#A78BFA,#22D3EE,#FBBF24,#FF6EC7);
       background-size:300% 100%;animation:flowBorder 5s linear infinite;"></div>

  <div class="sticker" style="top:14px; left:24px; animation-delay:0s;">✨</div>
  <div class="sticker" style="top:36px; right:40px; animation-delay:0.6s;">🌈</div>
  <div class="sticker" style="bottom:22px; left:48px; animation-delay:1.2s;">💾</div>
  <div class="sticker" style="bottom:30px; right:28px; animation-delay:1.8s;">👾</div>

  <div style="position:relative;z-index:2;height:100%;display:flex;flex-direction:column;
       align-items:center;justify-content:center;text-align:center;padding:0 20px;">
    <div style="font-size:2.9rem;font-weight:800;color:#312E81;letter-spacing:-0.5px;
         text-shadow:2px 2px 0 rgba(255,255,255,0.6);">
      🧠 <span style="background:linear-gradient(135deg,#4F46E5,#EC4899,#22D3EE);background-size:200% 200%;
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
      animation:shimmerText 5s ease infinite;">NLP Toolkit</span>
    </div>
    <div style="font-family:'Fredoka','Inter',sans-serif;font-size:1.05rem;color:#4338CA;margin-top:8px;font-weight:500;">
      Advanced Natural Language Processing &mdash; Analyze, Understand, and Extract Insights
    </div>
  </div>
</div>
<style>
@keyframes flowBorder { 0% { background-position: 0% 0%; } 100% { background-position: 300% 0%; } }
@keyframes shimmerText { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@keyframes stickerFloat {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-8px) rotate(8deg); }
}
.sticker {
    position: absolute;
    font-size: 1.4rem;
    z-index: 2;
    animation: stickerFloat 3.4s ease-in-out infinite;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.08));
}
</style>
<script>
  const canvas = document.getElementById('particles');
  const ctx = canvas.getContext('2d');
  function resize() { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; }
  resize();
  window.addEventListener('resize', resize);

  const colors = ['#FF9ED8', '#A5B4FC', '#67E8F9', '#FDE68A'];
  const particles = [];
  for (let i = 0; i < 46; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 2.2 + 1,
      dx: (Math.random() - 0.5) * 0.35,
      dy: (Math.random() - 0.5) * 0.35,
      c: colors[Math.floor(Math.random() * colors.length)]
    });
  }

  let mouseX = -999, mouseY = -999;
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = e.clientX - rect.left;
    mouseY = e.clientY - rect.top;
  });
  canvas.addEventListener('mouseleave', () => { mouseX = -999; mouseY = -999; });

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of particles) {
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.dy *= -1;

      const dMouse = Math.hypot(p.x - mouseX, p.y - mouseY);
      if (dMouse < 70) {
        const ang = Math.atan2(p.y - mouseY, p.x - mouseX);
        p.x += Math.cos(ang) * 1.1;
        p.y += Math.sin(ang) * 1.1;
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.c;
      ctx.globalAlpha = 0.75;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        if (dist < 85) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = 'rgba(167,139,250,0.16)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(animate);
  }
  animate();
</script>
"""
components.html(HERO_HTML, height=245)

# --------------------------------------------------
# INFO BADGES
# --------------------------------------------------

st.markdown(f"""
<div class="badge-row">
    <span class="info-badge">👤 Developer: {DEVELOPER_NAME}</span>
    <span class="info-badge">🗓️ Date: {PROJECT_DATE}</span>
    <span class="info-badge">🔢 Roll No: {ROLL_NO}</span>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown('<div class="section-label">Your Text</div>', unsafe_allow_html=True)

text = st.text_area(
    "📝 Enter your text for analysis",
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

    metrics_data = [
        ("📝", "Total Words", len(words)),
        ("📑", "Sentences", len(sentences)),
        ("🧹", "After Stopwords", len(filtered_words)),
        ("🏷️", "Named Entities", len(entities)),
        ("🔗", "Noun Phrases", len(chunks)),
        ("✨", "Unique Words", unique_words),
    ]

    METRICS_CARD_TEMPLATE = """
    <div class="metric-card" data-value="{value}">
        <div class="metric-icon">{icon}</div>
        <p class="metric-number">0</p>
        <p class="metric-label">{label}</p>
    </div>
    """
    metrics_cards_html = "".join(
        METRICS_CARD_TEMPLATE.format(value=value, icon=icon, label=label)
        for icon, label, value in metrics_data
    )

    METRICS_STYLE_AND_SCRIPT = """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Baloo+2:wght@700;800&display=swap" rel="stylesheet">
    <style>
        body { margin: 0; font-family: 'Inter', sans-serif; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 14px;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(253,251,255,0.9));
            border: 1px solid #E7E9F5;
            border-radius: 18px;
            padding: 18px 12px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            position: relative;
            overflow: hidden;
            transition: box-shadow 0.25s ease, border-color 0.25s ease;
            will-change: transform;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #FF6EC7, #A78BFA, #22D3EE);
            opacity: 0;
            transition: opacity 0.25s ease;
        }
        .metric-card:hover::before { opacity: 1; }
        .metric-card:hover {
            box-shadow: 0 12px 26px rgba(236, 110, 199, 0.16);
            border-color: #F5D0E8;
        }
        .metric-icon { font-size: 1.3rem; margin-bottom: 2px; }
        .metric-number {
            font-family: 'Baloo 2', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #FF6EC7 0%, #A78BFA 55%, #22D3EE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.2;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #64748B;
            margin-top: 0.2rem;
            font-weight: 500;
            letter-spacing: 0.3px;
        }
    </style>
    <script>
        const cards = document.querySelectorAll('.metric-card');
        cards.forEach((card) => {
            const target = parseInt(card.dataset.value, 10) || 0;
            const numEl = card.querySelector('.metric-number');
            const duration = 850;
            const startTime = performance.now();
            function step(ts) {
                const progress = Math.min((ts - startTime) / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                numEl.textContent = Math.round(eased * target);
                if (progress < 1) requestAnimationFrame(step);
            }
            requestAnimationFrame(step);

            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                const rotateX = (-y / rect.height) * 8;
                const rotateY = (x / rect.width) * 8;
                card.style.transform =
                    `perspective(600px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
            });
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'perspective(600px) rotateX(0) rotateY(0) translateY(0)';
            });
        });
    </script>
    """

    components.html(
        f'<div class="metrics-grid">{metrics_cards_html}</div>' + METRICS_STYLE_AND_SCRIPT,
        height=190,
    )
    
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
