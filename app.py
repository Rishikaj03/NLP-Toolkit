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
           Google Fonts
        ========================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap');

        /* ==========================================================
           Global
        ========================================================== */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 15.5px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }

        p, li, span, div {
            letter-spacing: 0.1px;
        }

        :root {
            --accent-1: #4F46E5;
            --accent-2: #0D9488;
            --accent-3: #D97706;
            --ink: #1E293B;
            --muted: #64748B;
            --line: #E2E8F0;
        }

        /* ==========================================================
           Background - light, airy, faint warmth (not plain white,
           not dim)
        ========================================================== */
        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -5%, rgba(79,70,229,0.045), transparent 60%),
                radial-gradient(1000px 500px at 95% 10%, rgba(13,148,136,0.045), transparent 60%),
                linear-gradient(180deg, #FBFAFF 0%, #FAFDFC 45%, #FDFCF9 100%);
            background-attachment: fixed;
        }

        .stApp::before, .stApp::after {
            content: '';
            position: fixed;
            border-radius: 50%;
            filter: blur(100px);
            z-index: 0;
            pointer-events: none;
            opacity: 0.25;
        }
        .stApp::before {
            width: 360px; height: 360px;
            top: -110px; left: -90px;
            background: radial-gradient(circle, rgba(79,70,229,0.16), transparent 70%);
        }
        .stApp::after {
            width: 380px; height: 380px;
            bottom: -110px; right: -90px;
            background: radial-gradient(circle, rgba(13,148,136,0.16), transparent 70%);
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
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            position: relative;
            z-index: 1;
        }

        /* ==========================================================
           Header - clean white card, thin gradient top edge
        ========================================================== */
        .header-container {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(8px);
            padding: 2.6rem 2rem 2.1rem 2rem;
            border-radius: 20px;
            margin-bottom: 2.5rem;
            border: 1px solid var(--line);
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 12px 28px rgba(15, 23, 42, 0.05);
            position: relative;
            overflow: hidden;
            transition: box-shadow 0.3s ease;
        }

        .header-container:hover {
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 16px 36px rgba(79, 70, 229, 0.09);
        }

        .header-container::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, #4F46E5, #7C3AED, #0EA5E9, #F59E0B);
            background-size: 300% 100%;
            animation: borderFlow 6s linear infinite;
        }

        @keyframes borderFlow {
            0% { background-position: 0% 0%; }
            100% { background-position: 300% 0%; }
        }

        .header-title {
            font-family: 'Sora', 'Inter', sans-serif;
            color: var(--ink);
            font-size: 2.6rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
        }

        .header-title span {
            background: linear-gradient(135deg, #4F46E5, #7C3AED, #0EA5E9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header-subtitle {
            color: var(--muted);
            text-align: center;
            font-size: 1.05rem;
            margin-top: 0.5rem;
            font-weight: 400;
            letter-spacing: 0.2px;
            position: relative;
            z-index: 1;
        }

        /* ==========================================================
           Generic Card
        ========================================================== */
        .card {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(8px);
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid var(--line);
            box-shadow: 0 1px 2px rgba(15,23,42,0.03);
            margin-bottom: 1.5rem;
            transition: all 0.25s ease;
        }

        .card:hover {
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            border-color: #C7D2FE;
            transform: translateY(-2px);
        }

        /* ==========================================================
           Headings
        ========================================================== */
        h1, h2, h3 {
            color: var(--ink) !important;
            font-family: 'Sora', 'Inter', sans-serif !important;
        }

        /* ==========================================================
           Text Area
        ========================================================== */
        .stTextArea textarea {
            background: #FFFFFF !important;
            color: var(--ink) !important;
            border: 2px solid var(--line) !important;
            border-radius: 14px !important;
            padding: 18px !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }

        .stTextArea textarea:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.10), 0 4px 14px rgba(99, 102, 241, 0.06) !important;
        }

        .stTextArea textarea::placeholder {
            color: #94A3B8 !important;
            font-weight: 300;
        }

        /* ==========================================================
           Button - clean gradient, subtle lift
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 54px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            font-size: 16px;
            font-weight: 600;
            font-family: 'Sora', 'Inter', sans-serif;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
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
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(79, 70, 229, 0.32);
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        .stButton > button:active {
            transform: translateY(0px);
        }

        /* ==========================================================
           Metric Cards - light, crisp, animated top edge on hover
        ========================================================== */
        .metric-card {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(8px);
            padding: 20px;
            border-radius: 16px;
            border: 1px solid var(--line);
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
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
            background: linear-gradient(90deg, #4F46E5, #7C3AED, #0EA5E9);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .metric-card:hover::before {
            opacity: 1;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 26px rgba(79, 70, 229, 0.10);
            border-color: #C7D2FE;
        }

        .metric-number {
            font-family: 'Sora', sans-serif;
            font-size: 2.3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
            line-height: 1.2;
        }

        .metric-label {
            font-size: 0.82rem;
            color: var(--muted);
            margin-top: 0.35rem;
            font-weight: 500;
            letter-spacing: 0.3px;
        }

        /* ==========================================================
           Tabs
        ========================================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #F1F5F9;
            padding: 6px;
            border-radius: 14px;
            border: 1px solid var(--line);
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-size: 0.88rem;
            color: var(--muted);
            font-family: 'Inter', sans-serif;
            transition: all 0.22s ease;
            background: transparent;
            border: none;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.7);
            color: var(--ink);
        }

        .stTabs [aria-selected="true"] {
            background: #FFFFFF !important;
            color: #4F46E5 !important;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.12);
            font-weight: 600;
        }

        /* ==========================================================
           Result Boxes
        ========================================================== */
        .result-box {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(8px);
            padding: 1rem 1.25rem;
            border-radius: 12px;
            border-left: 4px solid #6366F1;
            border-top: 1px solid var(--line);
            border-right: 1px solid var(--line);
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.75rem;
            color: var(--ink);
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.2s ease;
        }

        .result-box:hover {
            border-left-color: #7C3AED;
            transform: translateX(4px);
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.08);
        }

        .token-box {
            display: inline-block;
            background: #F1F5F9;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 10px;
            border: 1px solid var(--line);
            font-size: 0.9rem;
            color: var(--ink);
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .token-box:hover {
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
            border-color: transparent;
            color: #FFFFFF;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
        }

        /* ==========================================================
           Quick Stats
        ========================================================== */
        .quick-stats {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(8px);
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border: 1px solid var(--line);
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }

        .quick-stats p {
            margin: 0.4rem 0;
            color: #475569;
            font-size: 0.95rem;
        }

        .quick-stats strong {
            color: #4F46E5;
            font-weight: 600;
        }

        /* ==========================================================
           Common Words Box
        ========================================================== */
        .common-words {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(8px);
            padding: 0.75rem 1.25rem;
            border-radius: 12px;
            margin-top: 0.75rem;
            border: 1px solid var(--line);
            color: var(--ink);
            font-size: 0.95rem;
        }

        .common-words strong {
            color: #7C3AED;
        }

        /* ==========================================================
           Dataframes
        ========================================================== */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--line);
        }

        /* ==========================================================
           Footer
        ========================================================== */
        .footer {
            text-align: center;
            color: #94A3B8;
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid var(--line);
            margin-top: 2rem;
        }

        .footer span {
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
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
            background: #C7D2FE;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #A5B4FC;
        }

        /* ==========================================================
           Download button accent
        ========================================================== */
        [data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #0EA5E9 0%, #4F46E5 100%) !important;
            box-shadow: 0 4px 14px rgba(14, 165, 233, 0.25) !important;
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
            h1,h2,h3{ color: var(--ink);}

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

HERO_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700;800&display=swap" rel="stylesheet">
<div id="hero" style="position:relative;width:100%;height:196px;border-radius:22px;overflow:hidden;
     background:linear-gradient(135deg,#EEF2FF 0%,#ECFDF5 55%,#FFFBEB 100%);
     border:1px solid #E2E8F0;box-shadow:0 1px 3px rgba(15,23,42,0.05),0 14px 30px rgba(79,70,229,0.08);
     font-family:'Sora','Inter',sans-serif;">
  <canvas id="particles" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
  <div style="position:absolute;top:0;left:0;right:0;height:4px;
       background:linear-gradient(90deg,#4F46E5,#0D9488,#D97706,#4F46E5);
       background-size:300% 100%;animation:flowBorder 6s linear infinite;"></div>
  <div style="position:relative;z-index:2;height:100%;display:flex;flex-direction:column;
       align-items:center;justify-content:center;text-align:center;padding:0 20px;">
    <div style="font-size:2.4rem;font-weight:700;color:#312E81;letter-spacing:-0.5px;">
      🧠 <span style="background:linear-gradient(135deg,#4F46E5,#0D9488);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;background-clip:text;">NLP Toolkit</span>
    </div>
    <div style="font-size:1rem;color:#475569;margin-top:6px;">
      Advanced Natural Language Processing &mdash; Analyze, Understand, and Extract Insights
    </div>
  </div>
</div>
<style>
@keyframes flowBorder { 0% { background-position: 0% 0%; } 100% { background-position: 300% 0%; } }
</style>
<script>
  const canvas = document.getElementById('particles');
  const ctx = canvas.getContext('2d');
  function resize() { canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; }
  resize();
  window.addEventListener('resize', resize);

  const colors = ['#818CF8', '#5EEAD4', '#FCD34D'];
  const particles = [];
  for (let i = 0; i < 42; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 2 + 1,
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
      ctx.globalAlpha = 0.6;
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
          ctx.strokeStyle = 'rgba(79,70,229,0.10)';
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
components.html(HERO_HTML, height=210)

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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700;800&display=swap" rel="stylesheet">
    <style>
        body { margin: 0; font-family: 'Inter', sans-serif; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 14px;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.8));
            border: 1px solid #E2E8F0;
            border-radius: 16px;
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
            background: linear-gradient(90deg, #4F46E5, #0D9488, #D97706);
            opacity: 0;
            transition: opacity 0.25s ease;
        }
        .metric-card:hover::before { opacity: 1; }
        .metric-card:hover {
            box-shadow: 0 12px 26px rgba(79, 70, 229, 0.12);
            border-color: #C7D2FE;
        }
        .metric-icon { font-size: 1.3rem; margin-bottom: 2px; }
        .metric-number {
            font-family: 'Sora', sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4F46E5 0%, #0D9488 100%);
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