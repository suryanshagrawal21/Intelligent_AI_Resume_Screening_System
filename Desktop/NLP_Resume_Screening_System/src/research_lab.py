import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def get_jaccard_similarity(set1, set2):
    """
    Calculates Jaccard Similarity (Intersection over Union).
    Used as the 'Traditional Keyword Matching' baseline.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def compare_algorithms(resume_text, jd_text, nlp_score):
    """
    Compares traditional Jaccard matching vs NLP Cosine matching.
    Returns the two scores for visualization.
    """
    # Simple tokenization for Jaccard
    r_tokens = set(resume_text.lower().split())
    j_tokens = set(jd_text.lower().split())
    
    jaccard_score = get_jaccard_similarity(r_tokens, j_tokens)
    
    return {
        "Keyword (Jaccard) Score": round(jaccard_score, 4),
        "NLP (Cosine) Score": round(nlp_score, 4),
        "Improvement": round(nlp_score - jaccard_score, 4)
    }

def detect_bias_entities(text):
    """
    Checks if Name/Gender/Age entities are still detectable.
    This demonstrates the component of the system that filters them out.
    """
    doc = nlp(text)
    biased_entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "DATE", "AGE", "GPE"]: # GPE location can sometimes bias
            biased_entities.append((ent.text, ent.label_))
            
    return biased_entities
