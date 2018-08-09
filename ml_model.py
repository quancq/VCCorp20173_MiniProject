import numpy as np
import matplotlib.pyplot as plt
import utils
from collections import Counter
from preprocessing import FeatureTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


class EnsembleModel:
    def __init__(self, scoring):
        self.scoring = scoring
        self.models = {}
        self.feature_transformer = FeatureTransformer()

    def add_model(self, name, estimator):
        self.models.update({
            name: {
                "estimator": estimator,
                "pred": []
            }
        })

    def fit(self, X, y):
        # Transform raw document to document presentation
        X = self.feature_transformer.fit_transform(X, y)
        for name, model in self.models.items():
            model["estimator"].fit(X, y)
            print("Model {} fit done".format(name))
        self.print_stat_fit()

    def predict(self, X):
        X = self.feature_transformer.transform(X)
        total_preds = []
        for name, model in self.models.items():
            model["pred"] = model["estimator"].predict(X)
            for i, pred in enumerate(model["pred"]):
                total_preds[i].append(pred)

        # Major voting
        self.major_votings = []
        for i, preds in enumerate(total_preds):
            self.major_votings[i], _ = Counter(preds).most_common(1)[0]

        return self.major_votings

    def print_stat_fit(self):
        print("\n===============================")
        print("Statistic : ")
        for name, model in self.models.items():
            instance = model["estimator"]
            print("Model : ", name)
            print("Best params : ", instance.best_params_)
            print("Best score  : {}".format(instance.best_score_))
            for score in self.scoring:
                print("Mean {} score : {}".format(score, instance.cv_results_["mean_test_{}".format(score)]))
        print("===============================\n")


if __name__ == "__main__":
    training_data_path = "./Dataset/data_train.json"
    training_data = utils.load_data(training_data_path)
    X_train, y_train = utils.convert_orginal_data_to_list(training_data)

    scoring = ["f1_macro", "accuracy"]
    cv = 2
    model = EnsembleModel(scoring)

    # Multinomial Naive Bayes
    mnb_gs = GridSearchCV(
        MultinomialNB(),
        param_grid={"alpha": np.arange(0.8, 1, 0.3)},
        scoring=scoring,
        refit=scoring[0],
        cv=cv,
        return_train_score=True
    )
    model.add_model("MultinomialNB", mnb_gs)

    # Random Forest
    rf_gs = GridSearchCV(
        RandomForestClassifier(),
        param_grid={
            "max_features": np.linspace(0.6, 1, 1),
            "n_estimators": np.arange(15, 30, 40),
            # "min_samples_leaf": np.arange(2, 20, 5),
            # "max_depth": np.arange(30, 80, 30)
        },
        scoring=scoring,
        refit=scoring[0],
        cv=cv,
        return_train_score=True
    )
    model.add_model("RandomForest", rf_gs)

    model.fit(X_train, y_train)
