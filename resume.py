import streamlit as st
import PyPDF2
from google import genai # Ensure you have google-genai installed


client = genai.Client(api_key="gen-lang-client-0554766178")

def get_ai_feedback(jd, resume):
    prompt = f"""
    Analyze the following Job Description and Resume. 
    1. Provide 3 specific bullet points on how to improve the resume for this specific job.
    2. Suggest 2-3 specific technical skills missing.
    3. Keep the tone professional and encouraging.

    JD: {jd}
    Resume: {resume}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return "Feedback currently unavailable. Please check your API key."

# --- UI LOGIC (Insert this inside your 'if analyze_click:' block) ---

if analyze_click:
    if resume_file and jd_text:
        with st.spinner("AI is reading your resume..."):
            # ... (your existing PDF extraction code) ...
            pdf_reader = PyPDF2.PdfReader(resume_file)
            resume_text = " ".join([page.extract_text() for page in pdf_reader.pages])

            # Get Dynamic Feedback from Gemini
            dynamic_feedback = get_ai_feedback(jd_text, resume_text)

            # Displaying the dynamic feedback in your Glass UI
            st.markdown("### 💡 AI Personalized Feedback")
            st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; border-left: 5px solid #60a5fa;">
                    {dynamic_feedback}
                </div>
            """, unsafe_allow_html=True)
            
            # ... (rest of your UI) ...
            
