"""Embedding support: Sentence-Transformers or TF-IDF fallback."""
import numpy as np

class TextEmbedder:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.st_model = None
        self.bow_vectorizer = None
        
        # Try to load Sentence-Transformers if available
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer(model_name)
                print(f"✓ Loaded Sentence-Transformers model: {model_name}")
            except ImportError:
                print("⚠ sentence-transformers not installed, falling back to TF-IDF")
            except Exception as e:
                print(f"⚠ Failed to load model {model_name}: {e}")
        
        # Fallback: TF-IDF BoW
        if self.st_model is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.bow_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
            print("✓ Using TF-IDF embeddings (fallback)")
    
    def fit_bow(self, corpus):
        """Fit BoW on corpus if using TF-IDF."""
        if self.bow_vectorizer is not None:
            self.bow_vectorizer.fit(corpus)
    
    def encode(self, texts):
        """Encode texts to embeddings."""
        if self.st_model is not None:
            return self.st_model.encode(texts, convert_to_numpy=True)
        elif self.bow_vectorizer is not None:
            return self.bow_vectorizer.transform(texts).toarray()
        else:
            raise ValueError("No embedding model available")
    
    def cosine_similarity(self, query_emb, doc_embs):
        """Compute cosine similarity between query and documents."""
        from sklearn.metrics.pairwise import cosine_similarity
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        if len(doc_embs.shape) == 1:
            doc_embs = doc_embs.reshape(1, -1)
        return cosine_similarity(query_emb, doc_embs).ravel()
