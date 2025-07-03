import streamlit as st
import fitz  # PyMuPDF
import joblib
import os
import sys
import base64
import nltk
from nltk.corpus import stopwords

# Setup project path to import utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.cleaner import clean_text
from utils.parser import extract_text_from_pdf

# Load model and vectorizer
model_path = os.path.join(project_root, "models", "classifier.pkl")
vec_path = os.path.join(project_root, "models", "vectorizer.pkl")
model = joblib.load(model_path)
vectorizer = joblib.load(vec_path)

# Load Job Descriptions
jd_folder = os.path.join(project_root, "job_descriptions")
job_roles = {
    "data_science": "Data Scientist",
    "web_development": "Web Developer",
    "cloud_devops": "Cloud & DevOps Engineer",
    "embedded_engineer": "Embedded Engineer",
    "full_stack": "Full Stack Developer"
}

def load_jd(role_key):
    path = os.path.join(jd_folder, f"{role_key}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Job description not available."

def extract_keywords(text):
    nltk.download('stopwords', quiet=True)
    words = text.lower().split()
    stop_words = set(stopwords.words("english"))
    keywords = [w for w in words if w.isalpha() and w not in stop_words]
    return set(keywords)

def generate_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Prediction Report</a>'
    return href

# ------------------ 🔷 Streamlit Visual Styling ------------------

st.set_page_config(page_title="Smart Resume Screener", layout="centered", page_icon="📄")

st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #2E8B57;
        text-align: center;
        margin-top: 20px;
    }
    .sub-title {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        font-size: 14px;
        text-align: center;
        color: gray;
        padding-top: 30px;
        margin-top: 30px;
        border-top: 1px solid #eaeaea;
    }
    </style>
    <div class="main-title">📄 Smart Resume Screener</div>
    <div class="sub-title">Upload your resume and get a prediction of your ideal job role! 💼</div>
""", unsafe_allow_html=True)

# ------------------ 🔶 Resume Upload ------------------

uploaded_file = st.file_uploader("📎 Upload Your Resume (PDF format)", type="pdf")

if uploaded_file is not None:
    st.info("✅ Resume uploaded successfully!")

    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Extract and clean
        raw_text = extract_text_from_pdf("temp_resume.pdf")
        cleaned_text = clean_text(raw_text)
        vect_text = vectorizer.transform([cleaned_text])
        predicted_label = model.predict(vect_text)[0]
        confidence_score = model.predict_proba(vect_text).max()

        # ------------------ ✅ Show Prediction ------------------
        st.success(f"🎯 **Predicted Job Role:** {job_roles[predicted_label]}")
        st.markdown("### 📊 Confidence Score")
        st.progress(confidence_score)

        # ------------------ 🧠 Keyword Matching ------------------
        resume_keywords = extract_keywords(cleaned_text)
        jd_keywords = extract_keywords(load_jd(predicted_label))
        common_keywords = resume_keywords.intersection(jd_keywords)

        st.markdown("### 🧠 Matching Keywords Between Resume & JD")
        if common_keywords:
            st.success(f"Top matched skills: {', '.join(list(common_keywords)[:10])}")
        else:
            st.warning("No matching keywords found between resume and JD.")

        # ------------------ 📦 Download Report ------------------
        report_text = f"""
        🔍 Resume Screening Report
        
        Predicted Job Role: {job_roles[predicted_label]}
        Confidence Score: {round(confidence_score * 100, 2)}%
        
        Matched Keywords: {', '.join(list(common_keywords)[:15])}
        """

        st.markdown("### 📄 Download Your Report")
        st.markdown(generate_download_link(report_text, "resume_analysis.txt"), unsafe_allow_html=True)

        # ------------------ 📋 Job Description ------------------
        st.markdown("### 📋 Matched Job Description:")
        st.code(load_jd(predicted_label), language="text")

    except Exception as e:
        st.error(f"❌ Error processing resume: {e}")

# ------------------ 🔽 Footer ------------------
st.markdown("""
    <div class="footer">
        Made with ❤️ by <strong>Sornambal</strong> |
        <a href="https://github.com/Sornambal" target="_blank">GitHub</a>
    </div>
""", unsafe_allow_html=True)
