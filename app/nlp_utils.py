import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def process_file_for_tfidf(text: str, top_n: int = 10):
    # remove apostrophes and hyphens
    text = text.replace("'", "").replace("-", "")
    lines = text.strip().split('\n')
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(lines)

    scores = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    
    top_indices = scores.argsort()[::-1][:top_n]
    top_terms = [(terms[i], float(scores[i])) for i in top_indices]
    return top_terms
