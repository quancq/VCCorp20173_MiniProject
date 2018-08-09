import numpy as np
import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf = TfidfVectorizer()

    def fit(self, X, y):
        self.tfidf.fit(X, y)
        return self

    def transform(self, X):
        return self.tfidf.transform(X)
