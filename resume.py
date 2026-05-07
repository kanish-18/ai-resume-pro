import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# --- INITIALIZE AI CLIENT ---
# This pulls the key safely from Streamlit Cloud Secrets
try:
    client = genai.Client(api_key=st.secrets["gen-lang-client-0554766178"])
except Exception as e:
    st.error("API Key missing! Please add GEMINI_API_KEY to Streamlit Secrets.")

st.set_page_config(page_title="AI RESUME ANALYZER v6.0", layout="wide")

# --- UI STYLING ---
st.markdown("""
    <style>
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
    .premium-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 40px;
        margin-top: 50px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    .shiny-text {
        background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3em; text-align: center; letter-spacing: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important; border: none !important;
        border-radius: 15px !important; padding: 15px !important;
        width: 100%; font-weight: bold !important;
    }
    .footer-text {
        text-align: center; color: rgba(255, 255, 255, 0.4);
        font-size: 0.8em; margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MAIN UI ---
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<p class="shiny-text">SMART CV ANALYZER</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p style="color: #60a5fa; font-weight: bold;">01. INPUT DATA</p>', unsafe_allow_html=True)
    jd_text = st.text_area("JD", placeholder="Paste job requirements...", height=250, label_visibility="collapsed")

with col2:
    st.markdown('<p style="color: #a78bfa; font-weight: bold;">02. UPLOAD ARTIFACT</p>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume", type="pdf", label_visibility="collapsed")
    st.write("<br>"*3, unsafe_allow_html=True)
    
    # Define the variable globally to avoid NameError
    analyze_click = st.button("Execute Analysis")

# --- LOGIC BLOCK ---
if analyze_click:
    if resume_file and jd_text:
        with st.spinner("AI Engine Processing..."):
            # 1. Extract Text
            pdf_reader = PyPDF2.PdfReader(resume_file)
            resume_text = " ".join([page.extract_text() for page in pdf_reader.pages])
            
            # 2. Match Score (TF-IDF)
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
            match_val = round(cosine_similarity(tfidf_matrix)[0][1] * 100, 2)
            
            # 3. Dynamic AI Feedback using Gemini
            prompt = f"""
            Analyze the JD and Resume. 
            1. Give a 2-sentence summary of the match.
            2. List 3 specific things missing in the resume.
            3. Recommend 2 alternative job titles this person fits.
            JD: {jd_text[:1000]}
            Resume: {resume_text[:1000]}
            """
            
            try:
                response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                ai_feedback = response.text
            except:
                ai_feedback = "AI Feedback currently unavailable. Please verify API key."

            # --- DISPLAY RESULTS ---
            st.markdown(f"""
                <div style="margin-top: 30px; padding: 20px; border-radius: 20px; background: rgba(96, 165, 250, 0.1); border: 1px solid rgba(96, 165, 250, 0.3); text-align: center;">
                    <p style="color: #60a5fa; margin: 0;">MATCH SCORE</p>
                    <h1 style="color: white; font-size: 3.5rem; margin: 0;">{match_val}%</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🛠 AI Insights & Feedback")
            st.info(ai_feedback)

    else:
        st.warning("Please provide both the Job Description and the Resume PDF.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
    <div class='footer-text'>
        © 2026 | Built by <b>KANISH</b> | AI & ML  
    </div>
""", unsafe_allow_html=True)
    
