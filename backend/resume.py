from flask import Flask, request, jsonify
import pdfplumber
import spacy
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Allow frontend (CORS)
from flask_cors import CORS
CORS(app)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Predefined skill set
SKILL_SET = {"python", "flask", "nlp", "tensorflow", "react", "django", "docker", "sql", "javascript" ,"html" ,"css" ,"mongodb" ,"c++"}

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.lower()

def extract_skills(text):
    """Extracts skills from text based on predefined skill set."""
    doc = nlp(text)
    found_skills = {token.text.lower() for token in doc if token.text.lower() in SKILL_SET}
    return list(found_skills)

def get_bert_embedding(text):
    """Generates a BERT embedding for given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

@app.route("/upload", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = "uploaded_resume.pdf"
    file.save(file_path)

    # Extract resume text
    resume_text = extract_text_from_pdf(file_path)

    # Extract skills
    resume_skills = extract_skills(resume_text)

    # Example job description
    job_description = """
    We are looking for a Software Engineer with expertise in Python, Flask, SQL, and JavaScript.
    Candidates with React experience will be preferred.
    """

    job_skills = extract_skills(job_description)

    matched_skills = set(resume_skills) & set(job_skills)
    missing_skills = set(job_skills) - set(resume_skills)
    skill_match_percentage = (len(matched_skills) / len(job_skills)) * 100 if job_skills else 0

    # Compute similarity score
    resume_embedding = get_bert_embedding(resume_text)
    job_embedding = get_bert_embedding(job_description)
    similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0] * 100

    # Return results
    return jsonify({
    "match_score": round(float(similarity_score), 2),  # Convert to Python float
    "skill_match": round(float(skill_match_percentage), 2),  # Convert to Python float
    "matched_skills": list(matched_skills),
    "missing_skills": list(missing_skills)
})


if __name__ == "__main__":
    app.run(debug=True)
