import numpy as np
import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from pyvi import ViTokenizer
from stop_words import STOP_WORDS
import string


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenize = ViTokenizer.tokenize

        self.tfidf = TfidfVectorizer(
            stop_words=list(STOP_WORDS)
        )

    def fit(self, X, y):
        # Tokenize
        X = self.preprocess(X)
        self.tfidf.fit(X, y)
        return self

    def transform(self, X):
        # Tokenize
        X = self.preprocess(X)
        return self.tfidf.transform(X)

    def get_vocab(self):
        return self.tfidf.vocabulary_

    def preprocess(self, docs):
        print("Start Preprocess {} docs ...".format(len(docs)))
        result = []
        for i, doc in enumerate(docs):
            if i == 774:
                print("\nDocs 774 : ", doc)
            doc = re.sub(r'[^\w\s]','',doc)
            new_doc = []
            for word in doc.split(" "):
                if len(word) < 80:
                    new_doc.append(word)
            doc = " ".join(new_doc)
            print("After regex : ", doc)
            result.append(self.tokenize(doc))
            print("Preprocess ({}/{}) done".format(i + 1, len(docs)))

        # docs = [self.tokenize(doc) for doc in docs]
        print("Preprocess {} docs done".format(len(docs)))
        return result

    # def preprocessor(self, doc):
    #     # Remove all non-alpha-numeric-space from document
    #     return re.sub("[^a-zA-Z0-9 ]", "", doc)


if __name__ == "__main__":
    training_data_path = "./Dataset/data_train.json"
    training_data = utils.load_data(training_data_path)[::3]
    X_train, y_train = utils.convert_orginal_data_to_list(training_data)

    ft = FeatureTransformer()
    # ft.fit(X_train, y_train)
    # vocab = ft.get_vocab()
    # utils.write_vocab(vocab, "./ExploreResult/vitoken_vocab.txt")

    X_train = ft.preprocess(X_train)
    print(X_train[:3])
