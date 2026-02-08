from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global model cache to avoid reloading
_sbert_model = None

def get_sbert_model():
    global _sbert_model
    if _sbert_model is None:
        try:
            logging.info("Loading Sentence-BERT model...")
            _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logging.error(f"Failed to load SBERT model: {e}")
            return None
    return _sbert_model

def calculate_similarity(resumes_clean, jd_clean):
    """
    Calculates TF-IDF cosine similarity between resumes and job description.
    """
    corpus = [jd_clean] + resumes_clean
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    
    similarity_scores = cosine_similarity(jd_vector, resume_vectors).flatten()
    return similarity_scores

def calculate_sbert_similarity(resumes_clean, jd_clean):
    """
    Calculates Semantic similarity using Sentence-BERT embeddings.
    """
    model = get_sbert_model()
    if not model:
        return [0.0] * len(resumes_clean) # Fallback if model fails
        
    # Encode JD
    jd_embedding = model.encode([jd_clean])
    
    # Encode Resumes
    resume_embeddings = model.encode(resumes_clean)
    
    # Cosine Similarity
    scores = cosine_similarity(jd_embedding, resume_embeddings).flatten()
    return scores

def find_missing_skills(resume_text, jd_text, all_skills):
    """
    Identifies skills present in JD but missing in Resume.
    """
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    missing = []
    
    # We scan for known skills in JD
    jd_skills = [skill for skill in all_skills if skill in jd_lower]
    
    for skill in jd_skills:
        if skill not in resume_lower:
            missing.append(skill)
            
    return list(set(missing))
