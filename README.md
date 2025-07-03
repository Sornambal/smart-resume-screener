# 📄 Smart Resume Screener

A Machine Learning–powered app that predicts the most suitable job role based on your resume — built using **Python**, **Scikit-learn**, and **Streamlit**.

---

## 🚀 Demo

📂 [Project GitHub Repository](https://github.com/Sornambal/smart-resume-screener)

---

## 🧠 Key Features

🔹 **Resume Parsing (PDF)**  
Extracts text content from resumes using PyMuPDF.

🔹 **Resume Cleaning & Vectorization**  
Preprocesses and vectorizes the resume for ML prediction.

🔹 **Job Role Classification**  
Predicts roles like:
- Data Scientist
- Web Developer
- Cloud & DevOps Engineer
- Embedded Engineer
- Full Stack Developer

🔹 **Keyword Matching**  
Highlights common skills between resume and job description.

🔹 **Confidence Score**  
Displays model confidence for prediction.

🔹 **Downloadable Report**  
Users can download a `.txt` report summarizing their prediction and matched keywords.

---

## 📁 Project Structure
smart-resume-screener/
├── app/ # Streamlit UI app

├── models/ # Trained ML model & vectorizer

├── utils/ # Text parsing and cleaning helpers

├── job_descriptions/ # Sample JDs for each role

├── resumes/ # Sample resumes by role

├── notebooks/ # EDA & training notebooks

├── main.py # Core runner script

├── requirements.txt

└── README.md


---

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- Scikit-learn
- PyMuPDF (fitz)
- NLTK
- Joblib

---

🌟 Star this repo if you like it!


## 🔧 How to Run Locally


   ```bash
   git clone https://github.com/Sornambal/smart-resume-screener.git
   cd smart-resume-screener
   pip install -r requirements.txt
   streamlit run app/app.py



