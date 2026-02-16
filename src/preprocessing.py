import spacy
import re
from collections import Counter

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model is not present, we will need to download it. 
    # This might happen if the download command hasn't run yet.
    # We will handle this in the main app or installation script.
    print("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None

def clean_text(text):
    """
    Cleans the input text: removes URLs, emails, special characters, and extra spaces.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove non-alphanumeric characters (keep basic punctuation for sentence structure if needed, but for BoW usually remove)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.lower()

def preprocess_text(text):
    """
    Tokenizes, lemmatizes, and removes stop words.
    """
    if not nlp:
        return text.split() # Fallback

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

# Common Tech Skills Database (Expanded)
COMMON_SKILLS = {
    "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "react", "angular", "vue",
    "node.js", "express", "django", "flask", "fastapi", "sql", "mysql", "postgresql", "mongodb", "aws", 
    "azure", "gcp", "docker", "kubernetes", "git", "linux", "machine learning", "deep learning", "nlp", 
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "tableau", 
    "power bi", "excel", "communication", "teamwork", "leadership", "agile", "scrum", "project management"
}

def extract_skills(text):
    """
    Extracts skills from text based on a predefined list and basic NER.
    """
    if not text:
        return []
    
    text_lower = text.lower()
    extracted_skills = []
    
    # Keyword matching
    for skill in COMMON_SKILLS:
        # Use regex to find whole words only
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            extracted_skills.append(skill)
            
    return list(set(extracted_skills))

def extract_entities(text):
    """
    Extracts entities (ORG, PERSON, GPE) using spaCy.
    """
    if not nlp:
        return {}
    
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE"]:
            entities.setdefault(ent.label_, []).append(ent.text)
            
    return entities
