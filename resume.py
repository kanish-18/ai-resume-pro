import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


try:

    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception:
    st.error("🔑 API Key Missing! Settings > Secrets-இல் GEMINI_API_KEY-ஐச் சேர்க்கவும்.")
    st.stop()


@st.cache_data(show_spinner=False)
def get_ai_feedback(jd, resume):
    prompt = f"""
    As an expert HR Recruiter, analyze the match in Tamil:
    1. Summary of match (2 sentences).
    2. 3 Specific Missing Skills.
    3. 2 Tips to improve resume.
    4. 3 Job title recommendations.
    
    JD: {jd[:1000]}
    Resume: {resume[:1000]}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


st.set_page_config(page_title="AI RESUME ANALYZER v6.0", layout="wide")

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
        border-radius: 30px; padding: 40px; margin-top: 20px;
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
    }
    </style>
    """, unsafe_allow_html=True)


st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<p class="shiny-text">SMART CV ANALYZER</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p style="color: #60a5fa; font-weight: bold;">01. வேலை விவரம் (JD)</p>', unsafe_allow_html=True)
    jd_input = st.text_area("JD", placeholder="Paste job requirements...", height=250, label_visibility="collapsed")

with col2:
    st.markdown('<p style="color: #a78bfa; font-weight: bold;">02. உங்கள் RESUME (PDF)</p>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume", type="pdf", label_visibility="collapsed")
    st.write("<br>"*2, unsafe_allow_html=True)
    ு
    analyze_btn = st.button("ஆராய்ந்து பார் (EXECUTE)")


if analyze_btn:
    if resume_file and jd_input:
        with st.spinner("AI உங்களை ஆய்வு செய்கிறது..."):
  
            pdf_reader = PyPDF2.PdfReader(resume_file)
            resume_text = " ".join([page.extract_text() for page in pdf_reader.pages])
            
          
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([jd_input, resume_text])
            match_val = round(cosine_similarity(tfidf_matrix)[0][1] * 100, 2)
            
   
            feedback = get_ai_feedback(jd_input, resume_text)

           
            st.markdown(f"""
                <div style="margin-top: 30px; padding: 20px; border-radius: 20px; background: rgba(96, 165, 250, 0.1); border: 1px solid rgba(96, 165, 250, 0.3); text-align: center;">
                    <p style="color: #60a5fa; margin: 0;">MATCH SCORE</p>
                    <h1 style="color: white; font-size: 4rem; margin: 0;">{match_val}%</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🛠 AI ஆலோசனை & முடிவுகள்")
            st.write(feedback)
    else:
        st.warning("Give JD and PDF .")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray; font-size:0.8em; margin-top:30px;'>© 2026 | Built by KANISH | AI & ML Specialist</p>", unsafe_allow_html=True)
