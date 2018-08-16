import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
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
import utils
import pandas as pd
from sklearn.metrics import get_scorer, make_scorer, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


if __name__ == "__main__":

    # # Compare result
    # predict_path = "./Dataset/data_predict.json"
    # test_path = "./Dataset/data_test.json"
    #
    # predict_data = utils.load_data(predict_path)
    # test_data = utils.load_data(test_path)
    #
    # predict_data = pd.DataFrame(predict_data)
    # test_data = pd.DataFrame(test_data)
    #
    # # print(predict_data.head())
    # # print("=====================")
    # # print(test_data.head())
    #
    # result = pd.merge(predict_data, test_data, on="id")
    # y_pred = result.label_x
    # y_true = result.label_y
    #
    # metrics = {
    #     "accuracy": accuracy_score(y_true, y_pred),
    #     "f1-macro": f1_score(y_true, y_pred, average="macro"),
    #     "precision-macro": precision_score(y_true, y_pred, average="macro"),
    #     "recall-macro": precision_score(y_true, y_pred, average="macro"),
    # }
    #
    # for metric_name, score in metrics.items():
    #     print("{}: {:.4f}".format(metric_name, score))

    # Plot compare two distribution
    train_data_path = "./Dataset/data_train.json"
    test_data_path = "./Dataset/data_test.json"

    training_data = utils.load_data(train_data_path)
    test_data = utils.load_data(test_data_path)

    training_data = pd.DataFrame(training_data)
    test_data = pd.DataFrame(test_data)

    cols = ["label", "rate"]
    training_labels = training_data.label.value_counts().sort_index().reset_index()
    training_labels.columns = cols
    training_labels.rate = training_labels.rate * 100 / training_labels.rate.sum()

    test_labels = test_data.label.value_counts().sort_index().reset_index()
    test_labels.columns = cols
    test_labels.rate = test_labels.rate * 100 / test_labels.rate.sum()

    # Merge two df
    result = pd.merge(training_labels, test_labels, on="label", how="outer").sort_values("label").fillna(0)
    print(result)

    result.label = result.label.astype("str")

    mpl.style.use("seaborn")
    ax = result.plot(kind="scatter", x="rate_x", y="rate_y")
    ax.set(xlabel="Training_Rate", ylabel="Test_Rate", title="Compare two label distribution")
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    first = x_min if x_min < y_min else y_min
    last = x_max if x_max > y_max else y_max
    point = [first, last]
    ax.plot(point, point)

    save_path = "./ExploreResult/Compare_Distribution.png"
    plt.savefig(save_path)
