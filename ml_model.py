import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils
from collections import Counter
import time, os, json
from datetime import datetime
from preprocessing import FeatureTransformer
from sklearn.externals import joblib
from hyper_parameters import CV, VOCAB_PATH, RANDOM_STATE, SCORING, LABELS

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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

    def fit(self, X, y, is_encoded_data=True):
        if not is_encoded_data:
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

    def evaluate(self, X_test, y_test, metrics):
        # Predict X_test
        major_pred, _ = self.predict(X_test)

        # Evaluate models on metrics
        result = []
        cf_mats = {}
        columns = sorted(list(metrics.keys()))
        for name, model in self.models.items():
            row = [name]
            y_pred = model["pred"]
            for metric_name in columns:
                metric_fn = metrics.get(metric_name).get("fn")
                metric_params = metrics.get(metric_name).get("params")
                print("Score : {}, Params : {}".format(metric_name, metric_params))
                if metric_params is None:
                    value_score = metric_fn(y_test, y_pred, LABELS)
                else:
                    value_score = metric_fn(y_test, y_pred, LABELS, **metric_params)
                row.append(value_score)
            result.append(row)

            # Calculate confusion matrix
            cf_mat = confusion_matrix(y_test, y_pred, labels=LABELS)
            cf_mats.update({name: cf_mat})

        # Evaluate ensemble model
        ensemble_model_name = "Ensemble"
        row = [ensemble_model_name]
        for metric_name in columns:
            metric_fn = metrics.get(metric_name).get("fn")
            metric_params = metrics.get(metric_name).get("params")
            if metric_params is None:
                value_score = metric_fn(y_test, y_pred, LABELS)
            else:
                value_score = metric_fn(y_test, y_pred, LABELS, **metric_params)
            row.append(value_score)
        result.append(row)
        cf_mats.update({ensemble_model_name: confusion_matrix(y_test, major_pred, labels=LABELS)})

        columns = ["Model"] + columns
        result = pd.DataFrame(result, columns=columns)

        return result, cf_mats

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
            print("Training time : {} seconds".format(model["training_time"]))
        print("===============================\n")

    def get_statistic_data(self):
        data_plot = []
        columns = ["Model", "Hyper_Parameter"] + self.scoring + ["Training_Time (Seconds)"]
        for name, model in self.models.items():
            row = [name]
            instance = model["estimator"]
            row.append(instance.best_params_)
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
        statistic_data = self.get_statistic_data()

        # Save statistic data
        statistic_save_dir = os.path.join(save_dir, "Statistic")
        utils.mkdirs(statistic_save_dir)
        result_save_path = os.path.join(statistic_save_dir, "result.csv")
        statistic_data.to_csv(result_save_path, index=False)

        # Plot and save figure
        data_plot = statistic_data.drop("Hyper_Parameter", axis=1)
        self.plot_result(data_plot, statistic_save_dir, is_plot=False)

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
                                legend=None, color='C1', figsize=(len(self.models) + 1, 4), width=0.3)
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
    training_data_path = "./Dataset/data_train.json"
    training_encoded_data_path = "./Dataset/encoded_training_data_5453.json"
    # training_data = utils.load_data(training_data_path)
    # X_train, y_train = utils.convert_orginal_data_to_list(training_data)
    X_train, y_train = FeatureTransformer.load_encoded_data(training_encoded_data_path)

    model = EnsembleModel(SCORING, VOCAB_PATH, CV)

    # 1. Multinomial Naive Bayes
    mnb_gs = GridSearchCV(
        MultinomialNB(),
        param_grid={"alpha": [0.004]},
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False
    )
    model.add_model("MultiNB", mnb_gs)

    # 2. Random Forest
    rf_rs = RandomizedSearchCV(
        RandomForestClassifier(),
        param_distributions={
            # "max_features": np.arange(0.6, 1, 0.1),
            # "n_estimators": np.arange(10, 40, 5),
            # # "min_samples_leaf": np.arange(2, 20, 5),
            # "max_depth": np.arange(50, 90, 5),
            # "class_weight": ["balanced"],
            "max_features": [0.8],
            "n_estimators": [20],
            # "min_samples_leaf": np.arange(2, 20, 5),
            "max_depth": [80],
            "class_weight": ["balanced"],
            "n_jobs": [-1]
        },
        n_iter=1,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("RandomForest", rf_rs)

    # 3. Extra tree
    et_rs = RandomizedSearchCV(
        ExtraTreesClassifier(),
        param_distributions={
            "n_estimators": [50],
            "max_features": [0.3],
            "max_depth": [70],
            "n_jobs": [-1]
        },
        n_iter=1,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("ExtraTree", et_rs)

    # AdaBoost
    # adb_rs = RandomizedSearchCV(
    #     estimator=AdaBoostClassifier(),
    #     param_distributions={
    #         "n_estimators": np.arange(10, 100, 10),
    #         "learning_rate": np.arange(0.1, 1, 0.1)
    #     },
    #     n_iter=1,
    #     scoring=SCORING,
    #     refit=SCORING[0],
    #     cv=CV,
    #     return_train_score=False,
    #     random_state=RANDOM_STATE
    # )
    # model.add_model("AdaBoost", adb_rs)

    # 4. LightGBM
    lgbm_rs = RandomizedSearchCV(
        estimator=LGBMClassifier(),
        param_distributions={
            # "n_estimators": np.arange(40, 80, 10),
            # "learning_rate": np.arange(0.1, 0.4, 0.1),
            # "max_depth": np.arange(30, 60, 10),
            "n_estimators": [60],
            "learning_rate": [0.2],
            "max_depth": [60],
        },
        n_iter=1,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("LightGBM", lgbm_rs)

    # 5. Linear SVM
    linear_svm_gs = GridSearchCV(
        estimator=LinearSVC(),
        param_grid={
            "C": [1.1845]
        },
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
    )
    model.add_model("LinearSVM", linear_svm_gs)

    # 6. Kernel SVM
    kernel_svm_rs = RandomizedSearchCV(
        estimator=SVC(),
        param_distributions={
            "C": [0.7],
            "gamma": [0.4],
            "kernel": ["rbf"]
        },
        n_iter=1,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("KernelSVM", kernel_svm_rs)

    # 7. Logistic Regression
    lr_gs = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid={
            "C": [1.83]
        },
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
    )
    model.add_model("Logistic", lr_gs)

    # 8. KNN
    knn_rs = RandomizedSearchCV(
        estimator=KNeighborsClassifier(),
        param_distributions={
            "n_neighbors": [9],
            "weights": ["distance"],
        },
        n_iter=1,
        scoring=SCORING,
        refit=SCORING[0],
        cv=CV,
        return_train_score=False,
        random_state=RANDOM_STATE
    )
    model.add_model("KNN", knn_rs)

    # Bagging Multi NB
    # bg_mnb_rs = RandomizedSearchCV(
    #     estimator=BaggingClassifier(),
    #     param_distributions={
    #         "base_estimator": [MultinomialNB()],
    #         "n_estimators": np.arange(5, 50, 5),
    #         "max_features": np.arange(0.2, 1, 0.1)
    #     },
    #     n_iter=5,
    #     scoring=SCORING,
    #     refit=SCORING[0],
    #     cv=CV,
    #     return_train_score=False,
    #     random_state=RANDOM_STATE
    # )
    # model.add_model("Bagging_MNB", bg_mnb_rs)

    # Train model
    model.fit(X_train, y_train)

    # Save model
    model.save_model()
