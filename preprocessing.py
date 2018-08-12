import numpy as np
import pandas as pd
import utils
from utils import MyEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import re, os, time
from pyvi import ViTokenizer
from stop_words import STOP_WORDS as vi_stop_words
from nltk.corpus import stopwords
import json
from collections import Counter
from scipy.sparse import csr_matrix, vstack

from hyper_parameters import MIN_OCCURRENCES_TOKEN, MIN_WORD_LENGTH, MAX_WORD_LENGTH, NUM_LABELS, VOCAB_PATH


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_occurrences_of_token=MIN_OCCURRENCES_TOKEN, max_labels_of_token=NUM_LABELS * 0.5):
        self.tokenize = ViTokenizer.tokenize
        # Retain alpha character, underscore and white space in documents
        self.re = re.compile(r"[^\w\s]|[0-9_]")
        self.min_occurrences_of_token = min_occurrences_of_token
        self.max_labels_of_token = max_labels_of_token
        self.en_stop_words = set(stopwords.words('english'))

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
        start_time = time.time()
        print("Start Preprocess {} docs ...".format(len(docs)))
        result = []
        for i, doc in enumerate(docs):
            doc = self.clean_doc(doc)
            result.append(self.tokenize(doc))
            if (i+1) % 300 == 0:
                print("Preprocess ({}/{}) docs done".format(i + 1, len(docs)))

        finish_time = time.time()
        print("Preprocess {} docs done. Time : {} seconds".format(len(docs), (finish_time - start_time)))
        return result

    def clean_doc(self, doc):
        # Remove special character and word too long and convert to lower case
        cleaned_doc = []
        doc = self.re.sub('', doc)
        for word in doc.strip().split():
            if MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH and word not in self.en_stop_words:
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
                    token not in vi_stop_words:
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

    def save_vocab(self, dir="./Vocabulary"):
        utils.mkdirs(dir)
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
                    token not in vi_stop_words:
                self.vocab.update({token: i})

        print("Load vocabulary (size = {}) from {} done".format(len(self.vocab), path))
        return self.vocab

    def print_stats_vocab(self, num_records=5):
        print("\n Statistic full vocabulary - top ", num_records)
        print(self.vocab_stats_df.head(num_records))
        print("Mean number occurrences of each token : ", self.vocab_stats_df["Num_Occurrences"].mean())
        print("Mean number labels of each token      : ", self.vocab_stats_df["Num_Labels"].mean())

        print("\nFull vocabulary size : ", self.vocab_stats_df.shape[0])
        print("Use vocabulary size    : ", len(self.vocab))
        # print("Tf-idf vocabulary size : ", len(self.get_tfidf_vocab()))

        # print("Và :", self.vocab.get("và"), self.vocab.get("Và"))
        # print("Là :", self.vocab.get("là"), self.vocab.get("Là"))
        # print("Được :", self.vocab.get("được"), self.vocab.get("ĐưỢc"))

    @staticmethod
    def save_encoded_data(encoded_docs, labels, save_path):
        print("Start to save encoded data to ", save_path)
        data = []
        for doc, label in zip(encoded_docs, labels):
            data.append({
                "label": label,
                "encoded_content": {
                    "data": doc.data,
                    "indices": doc.indices,
                    "indptr": doc.indptr,
                    "shape": (doc.shape[0], doc.shape[1]),
                }
            })
        with open(save_path, 'w') as f:
            json.dump(data, f, cls=MyEncoder)
        print("Save encoded data (size = {}) to {} done".format(len(data), save_path))

    @staticmethod
    def load_encoded_data(save_path):
        print("Start to load encoded data from ", save_path)
        with open(save_path, 'r') as f:
            data = json.load(f)

        encoded_docs = []
        labels = []
        for elm in data:
            labels.append(elm["label"])
            encoded_content = elm["encoded_content"]
            en_data = encoded_content["data"]
            en_indices = encoded_content["indices"]
            en_indptr = encoded_content["indptr"]
            en_shape = encoded_content["shape"]
            sparse_content = csr_matrix((en_data, en_indices, en_indptr), shape=en_shape)
            encoded_docs.append(sparse_content)

        print("Load encoded data (size = {}) from {} done".format(len(labels), save_path))
        return vstack(encoded_docs), labels


if __name__ == "__main__":
    training_data_path = "./Dataset/data_train.json"
    training_data = utils.load_data(training_data_path)
    X_train, y_train = utils.convert_orginal_data_to_list(training_data)

    # Generate new vocabulary
    # min_occurrences_of_token, max_labels_of_token = 3, NUM_LABELS * 0.5
    # ft = FeatureTransformer(min_occurrences_of_token, max_labels_of_token)
    # ft.fit(X_train, y_train)
    # vocab_dir = "./Vocabulary/"
    # ft.save_vocab(vocab_dir)
    # ft.print_stats_vocab(10)
    # vocab = ft.get_tfidf_vocab()
    # utils.write_vocab(vocab, "./ExploreResult/vitoken_vocab_{}.txt".format(len(vocab)))

    # Save encoded documents
    ft = FeatureTransformer()
    X_train_encoded = ft.fit_transform(X_train, y_train, vocab_path=VOCAB_PATH)

    X_before = X_train_encoded[:5]
    y_before = y_train[-5:]

    encoded_save_path = "./Dataset/encoded_training_data_{}.json".format(len(y_train))
    FeatureTransformer.save_encoded_data(X_train_encoded, y_train, encoded_save_path)

    X_train_encoded_new, y_train_new = FeatureTransformer.load_encoded_data(encoded_save_path)
    X_after = X_train_encoded_new[:5]
    y_after = y_train_new[-5:]

    print("Compare : ", np.sum(X_before != X_after), np.sum(y_before != y_after))
    # print(X_before)

    print("\n\n===============")
    # print(X_after)

