import os
from utils.parser import extract_text_from_pdf
from utils.cleaner import clean_text
from utils.matcher import match_score

# ---- USER INPUT SECTION ----

# 1. Choose job role folder and resume path
job_role = "data_scientist"  # Options: data_scientist, web_developer, cloud_devops, embedded_engineer, full_stack
resume_filename = "sample1.pdf"  # File inside resumes/<job_role>/ directory

resume_path = os.path.join("resumes", job_role, resume_filename)
jd_path = os.path.join("job_descriptions", f"{job_role}.txt")

# ---- PIPELINE SECTION ----

# Step 1: Extract raw text from PDF
print(f"\n📄 Reading resume from: {resume_path}")
raw_text = extract_text_from_pdf(resume_path)

# Step 2: Clean and preprocess the text
print("🧹 Cleaning text...")
cleaned_text = clean_text(raw_text)

# Step 3: Load job description skill list
if not os.path.exists(jd_path):
    print(f"❌ Job description file not found at {jd_path}")
    exit()

with open(jd_path, "r") as file:
    jd_skills = [line.strip().lower() for line in file if line.strip()]

# Step 4: Match skills and calculate score
score, matched_skills = match_score(cleaned_text, jd_skills)

# ---- OUTPUT SECTION ----

print(f"\n🎯 Job Role Selected: {job_role.replace('_', ' ').title()}")
print(f"✅ Skills Matched: {score}/{len(jd_skills)}")
print(f"🔎 Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}")

# Optional: List missing skills
missing = list(set(jd_skills) - set([skill.lower() for skill in matched_skills]))
print(f"⚠️ Missing Skills: {', '.join(missing) if missing else 'None'}")
