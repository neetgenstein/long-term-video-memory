import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextIndex:
    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2))
        self.mat = self.vectorizer.fit_transform(texts)
        self.texts = texts

    def encode(self, texts):
        return self.vectorizer.transform(texts)

    def most_similar(self, query, topk=5):
        qv = self.encode([query])
        sims = cosine_similarity(qv, self.mat).ravel()
        idx = np.argsort(-sims)[:topk]
        return [(int(i), float(sims[i])) for i in idx]

class MultiTextIndex:
    def __init__(self, texts_by_type):
        # texts_by_type: dict[str, list[str]]
        self.indexes = {}
        for t, texts in texts_by_type.items():
            if texts:
                self.indexes[t] = TextIndex(texts)
            else:
                self.indexes[t] = None

    def sim(self, t, query, texts=None):
        idx = self.indexes.get(t)
        if not idx:
            return None
        qv = idx.encode([query])
        if texts is None:
            mat = idx.mat
        else:
            mat = idx.vectorizer.transform(texts)
        sims = cosine_similarity(qv, mat).ravel()
        return sims
