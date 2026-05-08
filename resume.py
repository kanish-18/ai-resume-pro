import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import time


try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("🔑 API Key Missing! Go to Settings > Secrets and add: GEMINI_API_KEY = 'your_key_here'")
    st.stop()

st.set_page_config(page_title="AI RESUME ANALYZER v6.0", layout="wide")

# --- 2. PREMIUM UI STYLING ---
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
        margin-top: 20px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    .shiny-text {
        background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3em; text-align: center; letter-spacing: 5px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important; border: none !important;
        border-radius: 15px !important; padding: 15px !important;
        width: 100%; font-weight: bold !important;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(139, 92, 246, 0.4); }
    .footer-text { text-align: center; color: rgba(255, 255, 255, 0.4); font-size: 0.8em; margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MAIN UI LAYOUT ---
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<p class="shiny-text">SMART CV ANALYZER</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p style="color: #60a5fa; font-weight: bold;">01. JOB DESCRIPTION</p>', unsafe_allow_html=True)
    jd_text = st.text_area("JD", placeholder="Paste job requirements...", height=250, label_visibility="collapsed")

with col2:
    st.markdown('<p style="color: #a78bfa; font-weight: bold;">02. UPLOAD RESUME (PDF)</p>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume", type="pdf", label_visibility="collapsed")
    st.write("<br>"*2, unsafe_allow_html=True)
    
    # Correct placement to avoid NameError
    analyze_click = st.button("EXECUTE AI ANALYSIS")

# --- 4. PROCESSING LOGIC ---
if analyze_click:
    if resume_file and jd_text:
        with st.spinner("🚀 Elite AI Engine Analyzing..."):
            try:
                # Extract PDF Text
                pdf_reader = PyPDF2.PdfReader(resume_file)
                resume_text = " ".join([page.extract_text() for page in pdf_reader.pages])
                
                # NLP Similarity Score
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
                match_val = round(cosine_similarity(tfidf_matrix)[0][1] * 100, 2)
                
                # Dynamic Gemini AI Feedback
                prompt = f"""
                As an expert HR Recruiter, compare this Resume against the JD.
                1. Summarize the match in 2 sentences.
                2. List 3 'Missing Skills' found in JD but not in Resume.
                3. Provide 2 specific tips to improve the Match Score.
                4. Recommend 3 job titles that fit this resume.
                
                JD: {jd_text}
                Resume: {resume_text}
                """
                
                response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                ai_feedback = response.text

                # --- 5. DISPLAY RESULTS ---
                st.markdown(f"""
                    <div style="margin-top: 30px; padding: 25px; border-radius: 20px; background: rgba(96, 165, 250, 0.1); border: 1px solid rgba(96, 165, 250, 0.3); text-align: center;">
                        <p style="color: #60a5fa; margin: 0; font-weight: bold; letter-spacing: 2px;">MATCH SCORE</p>
                        <h1 style="color: white; font-size: 4rem; margin: 0; text-shadow: 0 0 20px rgba(96, 165, 250, 0.5);">{match_val}%</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🛠 AI Career Strategy & Insights")
                st.markdown(f'<div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px;">{ai_feedback}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("⚠️ Please provide both the Job Description and the Resume PDF.")

st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(f"""
    <div class='footer-text'>
        © 2026 | Built by <b>KANISH</b> | AI & ML 
    </div>
""", unsafe_allow_html=True)
