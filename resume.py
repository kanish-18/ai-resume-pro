import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="AI RESUME ANALYZER v6.0", layout="wide")

st.markdown("""
    <style>
    /* Dynamic Mesh Gradient Background Animation */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #1e293b, #020617);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Ultra Glass Card with Motion */
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 40px;
        margin-top: 50px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .premium-card:hover {
        transform: scale(1.01);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Animated Header */
    .shiny-text {
        background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3em;
        text-align: center;
        letter-spacing: 10px;
        margin-bottom: 5px;
    }

    /* Button Motion */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px !important;
        width: 100%;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Footer Pulse */
    .footer-text {
        text-align: center;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.8em;
        margin-top: 50px;
        animation: pulse 3s infinite;
    }

    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 0.8; }
        100% { opacity: 0.4; }
    }

    /* Input Styling */
    .stTextArea textarea {
        background: rgba(0, 0, 0, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)



st.markdown('<div class="premium-card">', unsafe_allow_html=True)

st.markdown('<p class="shiny-text">Smart CV ANALYZER</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.5); letter-spacing: 3px; font-size: 0.7rem; margin-bottom: 40px;">AI-POWERED RESUME EXTRACTION ENGINE</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p style="color: #60a5fa; font-weight: bold; font-size: 0.8rem;">01. INPUT DATA</p>', unsafe_allow_html=True)
    jd_text = st.text_area("JD", placeholder="Paste job requirements...", height=250, label_visibility="collapsed")

with col2:
    st.markdown('<p style="color: #a78bfa; font-weight: bold; font-size: 0.8rem;">02. UPLOAD ARTIFACT</p>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume", type="pdf", label_visibility="collapsed")
    
    st.write("<br>"*3, unsafe_allow_html=True)
    analyze_click = st.button("Execute Analysis")

if analyze_click:
    if resume_file and jd_text:
        with st.spinner("Processing..."):
            # Extraction Logic
            pdf_reader = PyPDF2.PdfReader(resume_file)
            full_text = " ".join([page.extract_text() for page in pdf_reader.pages])
            
            # AI Logic
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([jd_text, full_text])
            match_val = round(cosine_similarity(tfidf_matrix)[0][1] * 100, 2)
            
            st.markdown(f"""
                <div style="margin-top: 30px; padding: 20px; border-radius: 20px; background: rgba(96, 165, 250, 0.1); border: 1px solid rgba(96, 165, 250, 0.3); text-align: center;">
                    <p style="color: #60a5fa; margin: 0;">MATCH SCORE</p>
                    <h1 style="color: white; font-size: 3.5rem; margin: 0;">{match_val}%</h1>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please provide both the Job Description and the Resume PDF.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
    <div class='footer-text'>
        © 2026 | Built by <b>KANISH</b> | AI & ML  
    </div>
""", unsafe_allow_html=True)
