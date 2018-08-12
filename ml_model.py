import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils
from collections import Counter
import time, os, json
from datetime import datetime
from preprocessing import FeatureTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import warnings

SCORING = ["f1_macro", "f1_micro", "accuracy"]
CV = 2
RANDOM_STATE = 7
VOCAB_PATH = "./Vocabulary/vocab_17012.csv"


class EnsembleModel:
    def __init__(self, scoring, vocab_path, cv=3):
        self.cv = cv
        self.scoring = scoring
        self.models = {}
        self.vocab_path = vocab_path
        self.feature_transformer = FeatureTransformer()

    def add_model(self, name, estimator):
        self.models.update({
            name: {
                "estimator": estimator,
                "pred": [],
                "training_time": 0
            }
        })

    def fit(self, X, y):
        # Transform raw document to document presentation
        X = self.feature_transformer.fit_transform(X, y, vocab_path=self.vocab_path)
        self.vocab = self.feature_transformer.get_tfidf_vocab()
        print("Vocabulary size : ", len(self.vocab))

        for name, model in self.models.items():
            start_time = time.time()
            model["estimator"].fit(X, y)
            finish_time = time.time()
            training_time = finish_time - start_time
            model["training_time"] = training_time
            print("Model {} fit done. Time : {:.4f} seconds".format(name, training_time))
        self.print_stat_fit()

    def predict(self, X):
        start_time = time.time()
        X = self.feature_transformer.transform(X)
        total_preds = [[] for _ in range(X.shape[0])]
        for name, model in self.models.items():
            model["pred"] = model["estimator"].predict(X)
            for i, pred in enumerate(model["pred"]):
                total_preds[i].append(pred)

        # Major voting
        self.major_votings = []
        model_predict_rate = []
        for i, preds in enumerate(total_preds):
            major_label, num_model_predict_label = Counter(preds).most_common(1)[0]
            self.major_votings.append(major_label)
            model_predict_rate.append(num_model_predict_label / len(self.models))

        finish_time = time.time()
        print("Model predict {} docs done. Time : {:.4f} seconds".format(X.shape[0], finish_time - start_time))
        return self.major_votings, model_predict_rate

    def print_stat_fit(self):
        print("\n===============================")
        print("Statistic : ")
        for name, model in self.models.items():
            instance = model["estimator"]
            print("\nModel : ", name)
            print("Best params : ", instance.best_params_)
            print("Best valid {} score  : {}".format(self.scoring[0], instance.best_score_))
            best_index = instance.best_index_
            for score in self.scoring:
                if score != self.scoring[0]:
                    print("Mean valid {} score : {}".format(score, instance.cv_results_["mean_test_{}".format(score)][best_index]))
        print("===============================\n")

    def get_data_plot(self):
        data_plot = []
        columns = ["Model"] + self.scoring + ["Training_Time (Seconds)"]
        for name, model in self.models.items():
            row = [name]
            instance = model["estimator"]
            row.append(instance.best_score_)
            best_index = instance.best_index_
            for score in self.scoring:
                if score != self.scoring[0]:
                    row.append(instance.cv_results_["mean_test_{}".format(score)][best_index])
            row.append(model["training_time"])
            data_plot.append(row)

        data_plot = pd.DataFrame(data_plot, columns=columns)
        return data_plot

    def save_model(self, save_dir="./Model"):
        print("Start to save {} models to {} ...".format(len(self.models), save_dir))
        save_dir = os.path.join(save_dir, utils.get_format_time_now())
        utils.mkdirs(save_dir)
        meta_data = []

        for name, model in self.models.items():
            instance = model["estimator"]
            save_path = os.path.join(save_dir, "{}.joblib".format(name))
            joblib.dump(instance, save_path)
            meta_data.append({
                "model_name": name,
                "model_path": save_path,
                "model_params": instance.best_params_
            })
            print("Save model {} to {} done".format(name, save_path))

        # Save meta data about models
        meta_data_path = os.path.join(save_dir, "meta.txt")
        # print("\nMeta data : ", meta_data)
        with open(meta_data_path, 'w') as f:
            json.dump(meta_data, f, cls=utils.MyEncoder)

        print("Save {} models to {} done".format(len(self.models), save_dir))

        # Save figure about training result of models
        # Create data frame contains result
        data_plot = self.get_data_plot()

        # Plot and save figure
        save_plot_dir = os.path.join(save_dir, "Statistic_Figure")
        self.plot_result(data_plot, save_plot_dir, is_plot=True)

    def load_model(self, save_dir):
        print("Start to load models from ", save_dir)
        meta_data_path = os.path.join(save_dir, "meta.txt")
        # Load meta data about models
        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)
        self.models = {}
        for info_model in meta_data:
            model_name = info_model["model_name"]
            model_path = info_model["model_path"]
            estimator = joblib.load(model_path)
            self.models.update({
                model_name: {
                    "estimator": estimator,
                    "pred": []
                }
            })
        self.feature_transformer.fit([""], [""], vocab_path=self.vocab_path)
        print("Load {} models from {} done".format(len(self.models), save_dir))

    def plot_result(self, data_plot, save_fig_dir, is_plot=True):
        utils.mkdirs(save_fig_dir)
        columns = list(data_plot.columns)
        print("Start to plot and save {} figures to {} ...".format(len(columns) - 1, save_fig_dir))

        print("Head of data plot")
        print(data_plot.head())
        x_offset = -0.07
        y_offset = 0.01
        mpl.style.use("seaborn")

        model_column = columns[0]
        for score_solumn in columns[1:]:
            # Sort by ascending score
            data_plot.sort_values(score_solumn, ascending=True, inplace=True)

            ax = data_plot.plot(kind="bar", x=model_column, y=score_solumn,
                                legend=None, color='C1', figsize=(len(self.models), 4), width=0.3)
            title = "Mean {} score - {} cross validation".format(score_solumn, self.cv)
            ax.set(title=title, xlabel=model_column, ylabel=score_solumn)
            ax.tick_params(axis='x', rotation=0)

            # Set lower and upper limit of y-axis
            min_score = data_plot.loc[:, score_solumn].min()
            max_score = data_plot.loc[:, score_solumn].max()
            y_lim_min = (min_score - 0.2) if min_score > 0.2 else 0
            y_lim_max = (max_score + 1) if max_score > 1 else 1
            ax.set_ylim([y_lim_min, y_lim_max])

            # Show value of each column to see clearly
            for p in ax.patches:
                b = p.get_bbox()
                text_value = "{:.4f}".format(b.y1)
                ax.annotate(text_value, xy=(b.x0 + x_offset, b.y1 + y_offset))

            save_fig_path = os.path.join(save_fig_dir, "{}.png".format(score_solumn))
            plt.savefig(save_fig_path, dpi=800)

        print("Plot and save {} figures to {} done".format(len(columns) - 1, save_fig_dir))
        if is_plot:
            plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("once")

    training_data_path = "./Dataset/data_train.json"
    training_data = utils.load_data(training_data_path)[:2000]
    X_train, y_train = utils.convert_orginal_data_to_list(training_data)

    model = EnsembleModel(SCORING, VOCAB_PATH, CV)

    # Multinomial Naive Bayes
    mnb_gs = GridSearchCV(
        MultinomialNB(),
        param_grid={"alpha": np.linspace(0.1, 1, 2)},
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False
    )
    model.add_model("MultiNB", mnb_gs)

    # Random Forest
    rf_rs = RandomizedSearchCV(
        RandomForestClassifier(),
        param_distributions={
            "max_features": np.linspace(0.1, 1, 10),
            "n_estimators": np.arange(15, 90, 20),
            # "min_samples_leaf": np.arange(2, 20, 5),
            "max_depth": np.arange(30, 80, 10)
        },
        n_iter=2,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("RandomForest", rf_rs)

    # Extra tree
    et_rs = RandomizedSearchCV(
        ExtraTreesClassifier(),
        param_distributions={
            "n_estimators": np.arange(10, 90, 20),
            # "min_samples_leaf": np.arange(2, 20, 5),
            "max_depth": np.arange(20, 80, 10)
        },
        n_iter=2,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("ExtraTree", et_rs)

    # AdaBoost
    adb_rs = RandomizedSearchCV(
        estimator=AdaBoostClassifier(),
        param_distributions={
            "n_estimators": np.arange(20, 80, 20),
            "learning_rate": np.arange(0.01, 1, 0.05)
        },
        n_iter=2,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("AdaBoost", adb_rs)

    # XGBoost
    xgb_rs = RandomizedSearchCV(
        estimator=XGBClassifier(),
        param_distributions={
            "n_estimators": np.arange(30, 120, 20),
            "learning_rate": np.arange(0.01, 1, 0.05),
            "max_depth": np.arange(5, 50, 10)
        },
        n_iter=2,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("XGBoost", xgb_rs)

    # Linear SVM
    linear_svm_rs = RandomizedSearchCV(
        estimator=LinearSVC(),
        param_distributions={
            "C": np.linspace(0.01, 1, 10)
        },
        n_iter=2,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("LinearSVM", linear_svm_rs)

    # Kernel SVM
    kernel_svm_rs = RandomizedSearchCV(
        estimator=SVC(),
        param_distributions={
            "C": np.arange(0.03, 1, 0.1),
            "gamma": np.arange(0.01, 1, 0.03),
            "kernel": ["rbf"]
        },
        n_iter=2,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("KernelSVM", kernel_svm_rs)

    # Logistic Regression
    lr_rs = RandomizedSearchCV(
        estimator=LogisticRegression(),
        param_distributions={
            "C": np.arange(0.01, 1, 0.05)
        },
        n_iter=2,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("Logistic", lr_rs)

    # Train model
    model.fit(X_train, y_train)

    # Save model
    model.save_model()
