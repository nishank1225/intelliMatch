import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NLPEngine:
    """
    NLP & Semantic Analysis Layer:
    Performs text preprocessing (cleaning, stopword removal, lemmatization simulation)
    and calculates semantic text similarity using TF-IDF and Cosine Similarity.
    """
    
    # Common standard stopwords
    STOPWORDS = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        self._fitted = False

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans raw text by removing punctuation, lowercasing, and removing stopwords.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        # Remove non-alphanumeric characters
        text = re.sub(r"[^\w\s]", " ", text)
        # Tokenize and remove stopwords
        tokens = [word for word in text.split() if word not in cls.STOPWORDS and len(word) > 1]
        
        # Basic rule-based lemmatization (stemming trailing s, ing, ed for similarity enhancement)
        cleaned_tokens = []
        for word in tokens:
            if word.endswith("ing") and len(word) > 5:
                word = word[:-3]
            elif word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("ed") and len(word) > 4:
                word = word[:-2]
            cleaned_tokens.append(word)
            
        return " ".join(cleaned_tokens)

    def fit_transform_corpus(self, text_list: list) -> np.ndarray:
        """
        Fits the TF-IDF vectorizer on the full corpus and transforms text to matrix vectors.
        """
        cleaned_corpus = [self.clean_text(t) for t in text_list]
        matrix = self.vectorizer.fit_transform(cleaned_corpus)
        self._fitted = True
        return matrix

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes cosine similarity between two single text fields.
        """
        c1 = self.clean_text(text1)
        c2 = self.clean_text(text2)
        if not c1 or not c2:
            return 0.0
        
        # Use vectorizer if fitted, else fit temporarily on pair
        if self._fitted:
            vecs = self.vectorizer.transform([c1, c2])
        else:
            temp_vec = TfidfVectorizer(ngram_range=(1, 2))
            vecs = temp_vec.fit_transform([c1, c2])
            
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
