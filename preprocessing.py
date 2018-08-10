import numpy as np
import pandas as pd
import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import re, os, time
from pyvi import ViTokenizer
from stop_words import STOP_WORDS
import string
from collections import Counter

MAX_WORD_LENGTH = 30
NUM_LABELS = 22


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_occurrences_of_token=3, max_labels_of_token=NUM_LABELS * 0.5):
        self.tokenize = ViTokenizer.tokenize
        # Retain alpha character, underscore and white space in documents
        self.re = re.compile(r"[^\w\s]|[0-9]")
        self.min_occurrences_of_token = min_occurrences_of_token
        self.max_labels_of_token = max_labels_of_token

    def fit(self, X, y, **fit_params):
        vocab_path = fit_params.get("vocab_path")
        # Build vocabulary from input or file
        if vocab_path is None:
            self.vocab = self.build_vocab(X, y, self.min_occurrences_of_token, self.max_labels_of_token)
        else:
            self.vocab = self.load_vocab(vocab_path, self.min_occurrences_of_token, self.max_labels_of_token)

        self.tfidf = TfidfVectorizer(
            # stop_words=list(STOP_WORDS),
            vocabulary=self.vocab
        )
        self.tfidf.fit(X, y)
        return self

    def transform(self, X):
        # Preprocess input
        X = self.preprocess(X)
        return self.tfidf.transform(X)

    def get_tfidf_vocab(self):
        return self.tfidf.vocabulary_

    def preprocess(self, docs):
        print("Start Preprocess {} docs ...".format(len(docs)))
        result = []
        for i, doc in enumerate(docs):
            doc = self.clean_doc(doc)
            result.append(self.tokenize(doc))
            if (i+1) % 300 == 0:
                print("Preprocess ({}/{}) docs done".format(i + 1, len(docs)))

        # docs = [self.tokenize(doc) for doc in docs]
        print("Preprocess {} docs done".format(len(docs)))
        return result

    def clean_doc(self, doc):
        # Remove special character and word too long and convert to lower case
        cleaned_doc = []
        doc = self.re.sub('', doc)
        for word in doc.strip().split():
            if len(word) < MAX_WORD_LENGTH:
                cleaned_doc.append(utils.lower(word))

        return " ".join(cleaned_doc)

    def build_vocab(self, docs, labels, min_occurrences_of_token, max_labels_of_token):
        start_time = time.time()
        print("Start build vocabulary from {} documents...".format(len(docs)))
        print("Min occurrence of token  : ", min_occurrences_of_token)
        print("Max topics of each token : ", max_labels_of_token)

        docs = self.preprocess(docs)

        token_counter = Counter()
        map_token_labels = {}

        for doc, label in zip(docs, labels):
            for token in doc.split():
                token_counter.update({token: 1})
                label_set = map_token_labels.get(token)
                if label_set is None:
                    label_set = set()
                    map_token_labels.update({token: label_set})
                label_set.add(label)

        # Filter token satisfy condition
        vocab = {}
        vocab_data = []
        vocab_size = 0
        for token in map_token_labels:
            num_labels_of_token = len(map_token_labels.get(token))
            num_occurrence_of_token = token_counter[token]

            # vocab_data.append((token, num_labels_of_token, num_occurrence_of_token))

            if num_labels_of_token <= max_labels_of_token and \
                    num_occurrence_of_token >= min_occurrences_of_token and \
                    token not in STOP_WORDS:
                vocab_data.append((token, num_labels_of_token, num_occurrence_of_token))
                vocab.update({token: vocab_size})
                vocab_size += 1

        # self.vocab = vocab

        columns = ["Token", "Num_Labels", "Num_Occurrences"]
        vocab_stats_df = pd.DataFrame.from_records(vocab_data, columns=columns)
        self.vocab_stats_df = vocab_stats_df

        finish_time = time.time()
        print("Build vocabulary (size = {}) done. Time : {:.4f} seconds".format(len(vocab), (finish_time - start_time)))

        return vocab

    def save_vocab(self, dir="./ExploreResult"):
        path = os.path.join(dir, "vocab_{}.csv".format(self.vocab_stats_df.shape[0]))
        self.vocab_stats_df.sort_values("Num_Occurrences", ascending=False, inplace=True)
        self.vocab_stats_df.to_csv(path, index=False)
        print("Save vocabulary to {} done".format(path))

    def load_vocab(self, path, min_occurrences_of_token, max_labels_of_token):
        self.vocab_stats_df = pd.read_csv(path)
        self.vocab = {}
        for i, row in self.vocab_stats_df.iterrows():
            token = row["Token"]
            num_labels_of_token = row["Num_Labels"]
            num_occurrence_of_token = row["Num_Occurrences"]
            if num_labels_of_token <= max_labels_of_token and \
                    num_occurrence_of_token >= min_occurrences_of_token and \
                    token not in STOP_WORDS:
                self.vocab.update({token: i})

        print("Load vocabulary (size = {}) from {} done".format(len(self.vocab), path))
        return self.vocab

    def print_stats_vocab(self, num_records=5):
        print("\n Statistic full vocabulary - top ", num_records)
        print(self.vocab_stats_df.head(num_records))
        print("\nFull vocabulary size : ", self.vocab_stats_df.shape[0])
        print("Use vocabulary size    : ", len(self.vocab))
        print("Tf-idf vocabulary size : ", len(self.get_tfidf_vocab()))

        # print("Và :", self.vocab.get("và"), self.vocab.get("Và"))
        # print("Là :", self.vocab.get("là"), self.vocab.get("Là"))
        # print("Được :", self.vocab.get("được"), self.vocab.get("ĐưỢc"))


if __name__ == "__main__":
    training_data_path = "./Dataset/data_train.json"
    training_data = utils.load_data(training_data_path)
    X_train, y_train = utils.convert_orginal_data_to_list(training_data)

    min_occurrences_of_token, max_labels_of_token = 3, NUM_LABELS * 0.5
    ft = FeatureTransformer(min_occurrences_of_token, max_labels_of_token)
    ft.fit(X_train, y_train)
    vocab_dir = "./ExploreResult/"
    ft.save_vocab(vocab_dir)
    ft.print_stats_vocab(10)
    # vocab = ft.get_tfidf_vocab()
    # utils.write_vocab(vocab, "./ExploreResult/vitoken_vocab_{}.txt".format(len(vocab)))

    # X_train = ft.preprocess(X_train)
    # print(X_train[:3])
