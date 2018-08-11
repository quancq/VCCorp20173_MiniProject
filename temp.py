import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import STOP_WORDS
from preprocessing import FeatureTransformer, NUM_LABELS
from pyvi import ViTokenizer
from nltk.corpus import stopwords


if __name__ == "__main__":
    stop = set(stopwords.words('english'))

    ft = FeatureTransformer()
    vocab_path = "./ExploreResult/vocab_17012.csv"
    vocab = ft.load_vocab(vocab_path, 3, NUM_LABELS * 0.5)
    ft.print_stats_vocab()
    # tokens_in_stopword = [token for token in vocab if token in stop]
    # for token in tokens_in_stopword:
    #     print(token)
    # print("Number tokens in stop word english : ", len(tokens_in_stopword))
