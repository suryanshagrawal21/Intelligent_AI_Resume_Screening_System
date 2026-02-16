import re
from collections import Counter

import spacy

# Load the spaCy language model for tokenisation and NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. "
          "Run: python -m spacy download en_core_web_sm")
    nlp = None


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text):
    """Strips URLs, emails, special chars, and extra whitespace."""
    if not text:
        return ""

    text = re.sub(r'http\S+', '', text)       # drop URLs
    text = re.sub(r'\S+@\S+', '', text)       # drop email addresses
    text = re.sub(r'[^\w\s]', ' ', text)       # keep only alphanumeric + spaces
    text = ' '.join(text.split())              # collapse whitespace

    return text.lower()


def preprocess_text(text):
    """Lemmatises tokens and removes stop words / punctuation."""
    if not nlp:
        return text.split()  # basic fallback when spaCy unavailable

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Skills Database & Extraction
# ---------------------------------------------------------------------------

# Curated list of common tech and soft skills for keyword matching
COMMON_SKILLS = {
    "python", "java", "c++", "c#", "javascript", "typescript", "html", "css",
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi",
    "sql", "mysql", "postgresql", "mongodb",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "tableau",
    "power bi", "excel",
    "communication", "teamwork", "leadership", "agile", "scrum", "project management",
}


def extract_skills(text):
    """Finds known skills in the text using whole-word regex matching."""
    if not text:
        return []

    text_lower = text.lower()
    found_skills = []

    for skill in COMMON_SKILLS:
        # \b ensures we match "python" but not "pythonic"
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)

    return list(set(found_skills))


# ---------------------------------------------------------------------------
# Named Entity Recognition
# ---------------------------------------------------------------------------

def extract_entities(text):
    """Pulls out organisations, people, and locations via spaCy NER."""
    if not nlp:
        return {}

    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE"]:
            entities.setdefault(ent.label_, []).append(ent.text)

    return entities
