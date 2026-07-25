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
           Color Variables - Mauve, Sky Blue, Royal Blue
        ========================================================== */
        :root {
            --mauve: #BCA4F5;
            --mauve-light: #D4C4F7;
            --mauve-dark: #A084E0;
            --sky-blue: #81CFFF;
            --sky-blue-light: #A8DEFF;
            --royal-blue: #4A69CE;
            --royal-blue-dark: #3A52A8;
            --royal-blue-light: #6A89E0;
            --honeydew: #E5F8F0;
            --tea-green: #ECFFBE;
            --bg-start: #F0F4FF;
            --bg-end: #F8F0FF;
        }

        /* ==========================================================
           Base
        ========================================================== */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 15px;
            line-height: 1.6;
        }

        /* ==========================================================
           Animated Background
        ========================================================== */
        .stApp {
            background: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%);
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
            animation: bgPulse 15s ease-in-out infinite;
            pointer-events: none;
        }

        @keyframes bgPulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
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
           Sidebar - Glassmorphism
        ========================================================== */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-right: 1px solid rgba(188, 164, 245, 0.2);
            padding: 1.5rem 1rem;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid rgba(188, 164, 245, 0.2);
        }

        .sidebar-logo {
            font-size: 2rem;
            background: linear-gradient(135deg, var(--royal-blue), var(--mauve));
            width: 48px;
            height: 48px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            animation: logoFloat 3s ease-in-out infinite;
        }

        @keyframes logoFloat {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-4px); }
        }

        .sidebar-title {
            font-weight: 700;
            font-size: 1.2rem;
            background: linear-gradient(135deg, var(--royal-blue), var(--mauve));
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
            color: var(--royal-blue);
            margin-bottom: 0.75rem;
        }

        .tech-tag {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            background: white;
            color: var(--royal-blue);
            border: 1px solid rgba(188, 164, 245, 0.3);
            transition: all 0.3s ease;
        }

        .tech-tag:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 105, 206, 0.15);
            border-color: var(--royal-blue);
        }

        .tech-tag.primary { background: rgba(74, 105, 206, 0.08); color: var(--royal-blue); }
        .tech-tag.pink { background: rgba(188, 164, 245, 0.08); color: var(--mauve-dark); }
        .tech-tag.cyan { background: rgba(129, 207, 255, 0.08); color: #3A8BC0; }
        .tech-tag.purple { background: rgba(188, 164, 245, 0.08); color: var(--mauve-dark); }

        .sidebar-footer {
            margin-top: auto;
            padding-top: 1rem;
            border-top: 1px solid rgba(188, 164, 245, 0.15);
            font-size: 0.75rem;
            color: #9CA3AF;
        }

        /* ==========================================================
           Hero - Glassmorphism with gradient
        ========================================================== */
        .hero-container {
            background: linear-gradient(135deg, var(--royal-blue) 0%, var(--mauve) 50%, var(--sky-blue) 100%);
            padding: 3rem 2.5rem 2.5rem 2.5rem;
            border-radius: 24px;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 15px 50px rgba(74, 105, 206, 0.2);
        }

        .hero-container::before {
            content: '✦';
            position: absolute;
            top: 10px;
            right: 30px;
            font-size: 80px;
            color: rgba(255, 255, 255, 0.04);
            animation: spin 20s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .hero-container::after {
            content: '✦';
            position: absolute;
            bottom: 10px;
            left: 30px;
            font-size: 60px;
            color: rgba(255, 255, 255, 0.03);
            animation: spin 15s linear infinite reverse;
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

        .hero-particles {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 0;
            pointer-events: none;
            overflow: hidden;
        }

        /* ==========================================================
           Cards - Glassmorphism
        ========================================================== */
        .card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 4px 20px rgba(74, 105, 206, 0.06);
            margin-bottom: 1.5rem;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(74, 105, 206, 0.1);
            border-color: rgba(188, 164, 245, 0.3);
        }

        /* ==========================================================
           Text Area
        ========================================================== */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(8px) !important;
            color: #1F2937 !important;
            border: 2px solid rgba(188, 164, 245, 0.2) !important;
            border-radius: 16px !important;
            padding: 18px !important;
            font-size: 15px !important;
            font-family: 'Inter', sans-serif !important;
            line-height: 1.7 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px rgba(74, 105, 206, 0.04) !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--royal-blue) !important;
            box-shadow: 0 0 0 4px rgba(74, 105, 206, 0.08), 0 4px 16px rgba(74, 105, 206, 0.06) !important;
            background: rgba(255, 255, 255, 0.95) !important;
        }

        /* ==========================================================
           Button - Animated Gradient
        ========================================================== */
        .stButton > button {
            width: 100%;
            height: 56px;
            border: none;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--royal-blue) 0%, var(--mauve) 50%, var(--sky-blue) 100%);
            background-size: 200% 200%;
            color: white;
            font-size: 16px;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 20px rgba(74, 105, 206, 0.25);
            letter-spacing: 0.5px;
            position: relative;
            overflow: hidden;
        }

        .stButton > button::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
            opacity: 0;
            transition: opacity 0.4s ease;
        }

        .stButton > button:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 8px 32px rgba(74, 105, 206, 0.35);
            background-position: 100% 100%;
        }

        .stButton > button:hover::before {
            opacity: 1;
        }

        /* ==========================================================
           Metric Cards - 3D Tilt Effect
        ========================================================== */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 14px;
            margin: 0.5rem 0 1rem 0;
        }

        .metric-item {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1.25rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(188, 164, 245, 0.15);
            text-align: center;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: default;
            position: relative;
            overflow: hidden;
        }

        .metric-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--royal-blue), var(--mauve), var(--sky-blue));
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .metric-item:hover::before {
            opacity: 1;
        }

        .metric-item:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 12px 40px rgba(74, 105, 206, 0.1);
            border-color: rgba(188, 164, 245, 0.3);
        }

        .metric-icon { font-size: 1.6rem; margin-bottom: 4px; display: block; }

        .metric-number {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--royal-blue), var(--mauve));
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
           Tabs - Modern Pill Style
        ========================================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 6px;
            border-radius: 16px;
            border: 1px solid rgba(188, 164, 245, 0.15);
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-size: 0.85rem;
            color: #6B7280;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            background: transparent;
            border: none;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255,255,255,0.6);
            color: var(--royal-blue);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--royal-blue), var(--mauve)) !important;
            color: white !important;
            box-shadow: 0 4px 16px rgba(74, 105, 206, 0.15);
            font-weight: 600;
        }

        /* ==========================================================
           Result Boxes
        ========================================================== */
        .result-box {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border-left: 4px solid var(--royal-blue);
            border-top: 1px solid rgba(188, 164, 245, 0.15);
            border-right: 1px solid rgba(188, 164, 245, 0.15);
            border-bottom: 1px solid rgba(188, 164, 245, 0.15);
            margin-bottom: 0.75rem;
            color: #1F2937;
            font-size: 0.95rem;
            line-height: 1.6;
            transition: all 0.3s ease;
        }

        .result-box:hover {
            background: rgba(255, 255, 255, 0.9);
            border-left-color: var(--mauve);
            transform: translateX(6px);
            box-shadow: 0 4px 16px rgba(74, 105, 206, 0.06);
        }

        /* ==========================================================
           Token Boxes
        ========================================================== */
        .token-box {
            display: inline-block;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 10px;
            border: 1px solid rgba(188, 164, 245, 0.15);
            font-size: 0.9rem;
            color: #1F2937;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .token-box:hover {
            background: rgba(74, 105, 206, 0.08);
            border-color: var(--royal-blue);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 105, 206, 0.08);
        }

        /* ==========================================================
           Quick Stats
        ========================================================== */
        .quick-stats {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1rem 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(188, 164, 245, 0.15);
            box-shadow: 0 2px 8px rgba(74, 105, 206, 0.04);
        }

        .quick-stats p {
            margin: 0.4rem 0;
            color: #4B5563;
            font-size: 0.95rem;
        }

        .quick-stats strong {
            color: var(--royal-blue);
            font-weight: 600;
        }

        .common-words {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 0.75rem 1.25rem;
            border-radius: 14px;
            margin-top: 0.75rem;
            border: 1px solid rgba(188, 164, 245, 0.15);
            color: #1F2937;
            font-size: 0.95rem;
        }

        .common-words strong {
            color: var(--royal-blue);
        }

        .common-words span {
            background: rgba(74, 105, 206, 0.06);
            padding: 0.2rem 0.8rem;
            border-radius: 12px;
            display: inline-block;
            margin: 0.15rem;
            border: 1px solid rgba(188, 164, 245, 0.1);
        }

        /* ==========================================================
           Footer
        ========================================================== */
        .footer {
            text-align: center;
            color: #9CA3AF;
            padding: 1.5rem 0 0.5rem 0;
            font-size: 0.85rem;
            border-top: 1px solid rgba(188, 164, 245, 0.15);
            margin-top: 2rem;
        }

        .footer span {
            background: linear-gradient(135deg, var(--royal-blue), var(--mauve));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 600;
        }

        /* ==========================================================
           Scrollbar
        ========================================================== */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: rgba(188, 164, 245, 0.1); border-radius: 10px; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, var(--royal-blue), var(--mauve));
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--royal-blue-dark); }

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
# HERO WITH INTERACTIVE PARTICLES
# --------------------------------------------------

st.markdown("""
<div class="hero-container">
    <div class="hero-particles" id="heroParticles"></div>
    <h1 class="hero-title">🧠 Natural Language <span>Processing</span></h1>
    <p class="hero-subtitle">Analyze, understand, and extract insights from your text with advanced NLP techniques</p>
</div>

<script>
    // Interactive particle animation for hero
    (function() {
        const container = document.getElementById('heroParticles');
        if (!container) return;
        
        const canvas = document.createElement('canvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        let width, height;
        const particles = [];
        const colors = ['rgba(255,255,255,0.06)', 'rgba(255,255,255,0.04)', 'rgba(255,255,255,0.08)'];
        
        function resize() {
            const rect = container.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            width = canvas.width;
            height = canvas.height;
        }
        
        resize();
        window.addEventListener('resize', resize);
        
        for (let i = 0; i < 40; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                r: Math.random() * 2 + 1,
                dx: (Math.random() - 0.5) * 0.5,
                dy: (Math.random() - 0.5) * 0.5,
                color: colors[Math.floor(Math.random() * colors.length)]
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
            ctx.clearRect(0, 0, width, height);
            
            for (const p of particles) {
                p.x += p.dx;
                p.y += p.dy;
                if (p.x < 0 || p.x > width) p.dx *= -1;
                if (p.y < 0 || p.y > height) p.dy *= -1;
                
                const dMouse = Math.hypot(p.x - mouseX, p.y - mouseY);
                if (dMouse < 100) {
                    const ang = Math.atan2(p.y - mouseY, p.x - mouseX);
                    p.x += Math.cos(ang) * 0.8;
                    p.y += Math.sin(ang) * 0.8;
                }
                
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                ctx.fill();
            }
            
            // Draw connections
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const a = particles[i], b = particles[j];
                    const dist = Math.hypot(a.x - b.x, a.y - b.y);
                    if (dist < 120) {
                        ctx.beginPath();
                        ctx.moveTo(a.x, a.y);
                        ctx.lineTo(b.x, b.y);
                        ctx.strokeStyle = `rgba(255,255,255,${0.03 * (1 - dist/120)})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }
            
            requestAnimationFrame(animate);
        }
        
        animate();
    })();
</script>
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
    # METRICS WITH 3D TILT EFFECT
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
        <div class="metric-item" data-value="{value}">
            <span class="metric-icon">{icon}</span>
            <div class="metric-number" data-target="{value}">0</div>
            <div class="metric-label">{label}</div>
        </div>
        """
    metrics_html += '</div>'
    
    metrics_html += """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Animate numbers
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
            
            // 3D Tilt Effect
            item.addEventListener('mousemove', (e) => {
                const rect = item.getBoundingClientRect();
                const x = e.clientX - rect.left - rect.width / 2;
                const y = e.clientY - rect.top - rect.height / 2;
                const rotateX = (-y / rect.height) * 8;
                const rotateY = (x / rect.width) * 8;
                item.style.transform = 
                    `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
            });
            item.addEventListener('mouseleave', () => {
                item.style.transform = 'perspective(800px) rotateX(0) rotateY(0) translateY(0)';
            });
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