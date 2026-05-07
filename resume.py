import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI RESUME ANALYZER v6.0", layout="wide")

# --- CUSTOM CSS (Kept your original + new styles) ---
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
    }
    .shiny-text {
        background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3em; text-align: center; letter-spacing: 10px;
    }
    .feature-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border-left: 5px solid #a78bfa;
        margin-bottom: 15px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important; border-radius: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def extract_skills(text):
    # Basic common skills list (In a real app, use a larger dataset or Spacy)
    skills_db = ["Python", "Java", "React", "SQL", "Machine Learning", "Cloud", "Project Management", "UI/UX", "Docker", "Kubernetes", "C++", "Excel", "Data Analysis"]
    found_skills = [skill for skill in skills_db if skill.lower() in text.lower()]
    return set(found_skills)

def get_job_recommendations(text):
    # Simple logic to recommend roles based on keywords
    text = text.lower()
    if "python" in text or "machine learning" in text:
        return ["Data Scientist", "AI Engineer", "Backend Developer"]
    elif "react" in text or "ui" in text:
        return ["Frontend Developer", "Product Designer", "Full Stack Engineer"]
    else:
        return ["General Consultant", "Project Coordinator", "Business Analyst"]

# --- UI LAYOUT ---
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown('<p class="shiny-text">SMART CV ANALYZER</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p style="color: #60a5fa; font-weight: bold;">01. JOB DESCRIPTION</p>', unsafe_allow_html=True)
    jd_text = st.text_area("JD", placeholder="Paste the job requirements here...", height=200, label_visibility="collapsed")

with col2:
    st.markdown('<p style="color: #a78bfa; font-weight: bold;">02. UPLOAD RESUME</p>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Resume", type="pdf", label_visibility="collapsed")
    analyze_click = st.button("Analyze Profile")

if analyze_click:
    if resume_file and jd_text:
        with st.spinner("Deep Scan in Progress..."):
            # 1. Extraction
            pdf_reader = PyPDF2.PdfReader(resume_file)
            resume_text = " ".join([page.extract_text() for page in pdf_reader.pages])
            
            # 2. Similarity Score
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
            match_val = round(cosine_similarity(tfidf_matrix)[0][1] * 100, 2)
            
            # 3. Logic for Features
            jd_skills = extract_skills(jd_text)
            resume_skills = extract_skills(resume_text)
            missing_skills = jd_skills - resume_skills
            recommendations = get_job_recommendations(resume_text)

            # --- DISPLAY RESULTS ---
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 30px;">
                    <h2 style="color: white;">Match Score: <span style="color: #60a5fa;">{match_val}%</span></h2>
                </div>
            """, unsafe_allow_html=True)

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("### 🛠 Skill Gap Analysis")
                if missing_skills:
                    st.write("Consider adding these skills to your resume:")
                    for skill in missing_skills:
                        st.markdown(f"- **{skill}**")
                else:
                    st.success("Your skills align perfectly with the JD!")

                st.markdown("### 💡 AI Feedback")
                if match_val < 50:
                    st.info("Tip: Your resume lacks many keywords found in the JD. Try tailoring your 'Experience' section to include specific tools mentioned.")
                else:
                    st.success("Strong match! Ensure your contact info is updated before applying.")

            with res_col2:
                st.markdown("### 🚀 Career Roadmap")
                st.write("Based on your profile, you are a great fit for:")
                for job in recommendations:
                    st.markdown(f"""
                        <div class="feature-box">
                            <b>{job}</b><br><small>Click to search on LinkedIn</small>
                        </div>
                    """, unsafe_allow_html=True)

    else:
        st.error("Missing Data: Please upload a PDF and paste a Job Description.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
    <div class='footer-text'>
        © 2026 | Built by <b>KANISH</b> | AI & ML  
    </div>
""", unsafe_allow_html=True)
            
