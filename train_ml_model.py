from hyper_parameters import CV, VOCAB_PATH, RANDOM_STATE, SCORING
import utils
from ml_model import EnsembleModel
from preprocessing import FeatureTransformer

import numpy as np

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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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
    # rf_rs = RandomizedSearchCV(
    #     RandomForestClassifier(),
    #     param_distributions={
    #         # "max_features": np.arange(0.6, 1, 0.1),
    #         # "n_estimators": np.arange(10, 40, 5),
    #         # # "min_samples_leaf": np.arange(2, 20, 5),
    #         # "max_depth": np.arange(50, 90, 5),
    #         # "class_weight": ["balanced"],
    #         "max_features": [0.8],
    #         "n_estimators": [20],
    #         # "min_samples_leaf": np.arange(2, 20, 5),
    #         "max_depth": [80],
    #         "class_weight": ["balanced"],
    #         "n_jobs": [-1]
    #     },
    #     n_iter=1,
    #     scoring=SCORING,
    #     refit=SCORING[0],
    #     cv=CV,
    #     return_train_score=False,
    #     random_state=RANDOM_STATE
    # )
    # model.add_model("RandomForest", rf_rs)

    # 3. Extra tree
    # et_rs = RandomizedSearchCV(
    #     ExtraTreesClassifier(),
    #     param_distributions={
    #         "n_estimators": [50],
    #         "max_features": [0.3],
    #         "max_depth": [70],
    #         "n_jobs": [-1]
    #     },
    #     n_iter=1,
    #     scoring=SCORING,
    #     refit=SCORING[0],
    #     cv=CV,
    #     return_train_score=False,
    #     random_state=RANDOM_STATE
    # )
    # model.add_model("ExtraTree", et_rs)

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
            # "n_estimators": np.arange(40, 70, 10),
            # "learning_rate": np.arange(0.15, 0.3, 0.05),
            # "max_depth": np.arange(40, 60, 10),
            "n_estimators": [50],
            "learning_rate": [0.2],
            "max_depth": [50],
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
    # linear_svm_gs = GridSearchCV(
    #     estimator=LinearSVC(),
    #     param_grid={
    #         "C": [1.1845]
    #     },
    #     scoring=SCORING,
    #     refit=SCORING[0],
    #     cv=CV,
    #     return_train_score=False,
    # )
    # model.add_model("LinearSVM", linear_svm_gs)

    # 6. Kernel SVM
    # kernel_svm_rs = RandomizedSearchCV(
    #     estimator=SVC(),
    #     param_distributions={
    #         "C": [0.7],
    #         "gamma": [0.4],
    #         "kernel": ["rbf"]
    #     },
    #     n_iter=1,
    #     scoring=SCORING,
    #     refit=SCORING[0],
    #     cv=CV,
    #     return_train_score=False,
    #     random_state=RANDOM_STATE
    # )
    # model.add_model("KernelSVM", kernel_svm_rs)

    # 7. Logistic Regression
    # lr_gs = GridSearchCV(
    #     estimator=LogisticRegression(),
    #     param_grid={
    #         "C": [1.83]
    #     },
    #     scoring=SCORING,
    #     refit=SCORING[0],
    #     cv=CV,
    #     return_train_score=False,
    # )
    # model.add_model("Logistic", lr_gs)

    # 8. KNN
    # knn_rs = RandomizedSearchCV(
    #     estimator=KNeighborsClassifier(),
    #     param_distributions={
    #         "n_neighbors": [9],
    #         "weights": ["distance"],
    #     },
    #     n_iter=1,
    #     scoring=SCORING,
    #     refit=SCORING[0],
    #     cv=CV,
    #     return_train_score=False,
    #     random_state=RANDOM_STATE
    # )
    # model.add_model("KNN", knn_rs)

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
