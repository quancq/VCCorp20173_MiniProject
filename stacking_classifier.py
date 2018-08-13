from sklearn import model_selection
from mlxtend.classifier import StackingClassifier
import numpy as np
import pandas as pd
from preprocessing import FeatureTransformer
import utils, os, time
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

from hyper_parameters import MAP_LABEL_ID_TO_TEXT
from sklearn.metrics import get_scorer, make_scorer, f1_score, precision_score, recall_score, accuracy_score
from hyper_parameters import LABELS, RANDOM_STATE, VOCAB_PATH

def f1_macro_score(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, labels, average="macro")


def f1_micro_score(y_true, y_pred, labels):
    return f1_score(y_true, y_pred, labels, average="micro")


def precision_macro_score(y_true, y_pred, labels):
    return precision_score(y_true, y_pred, labels, average="macro")


def precision_micro_score(y_true, y_pred, labels):
    return precision_score(y_true, y_pred, labels, average="micro")


def recall_macro_score(y_true, y_pred, labels):
    return recall_score(y_true, y_pred, labels, average="macro")


def recall_micro_score(y_true, y_pred, labels):
    return recall_score(y_true, y_pred, labels, average="micro")


if __name__ == "__main__":
    # Load trining data
    training_encoded_data_path = "./Dataset/encoded_training_data_4362.json"
    X_train, y_train = FeatureTransformer.load_encoded_data(training_encoded_data_path)

    # Load test data
    test_data_path = "./Dataset/valid_data_1091.json"
    test_data = utils.load_data(test_data_path)
    df = pd.DataFrame(test_data)
    X_test = df.content.values
    y_test = df.label.values

    # Transform test data
    ft = FeatureTransformer()
    X_test = ft.fit_transform(X_test, y_train, vocab_path=VOCAB_PATH)

    # Define models
    mnb = MultinomialNB(alpha=0.004)

    rf = RandomForestClassifier(
        max_features=0.8,
        n_estimators=20,
        max_depth=80,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE)

    etree = ExtraTreesClassifier(
        n_estimators=50,
        max_features=0.3,
        max_depth=70,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    linear_svm = LinearSVC(
        C=1.1845,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    knn = KNeighborsClassifier(
        n_neighbors=9,
        weights="distance"
    )

    models = {
        "MultiNB": mnb,
        "KNN": knn,
        "RandomForest": rf,
        "ExtraTree": etree,
        "LinearSVM": linear_svm
    }

    lr = LogisticRegression(
        C=1.83,
        random_state=RANDOM_STATE
    )

    params = {
        # 'kneighborsclassifier__n_neighbors': [1, 5],
        # 'randomforestclassifier__n_estimators': [10, 50],
        'meta-logisticregression__C': np.arange(0.1, 1, 0.3)
    }

    sclf = StackingClassifier(classifiers=list(models.values()), meta_classifier=lr)
    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=3,
                        refit="accuracy")

    models.update({"Stacking": grid})

    metrics = {
        "accuracy": accuracy_score,
        "f1_macro": f1_macro_score,
        "f1_micro": f1_micro_score,
        "precision_macro": precision_macro_score,
        "precision_micro": precision_micro_score,
        "recall_macro": recall_macro_score,
        "recall_micro": recall_micro_score
    }

    evaluate_result = []
    for model_name, clf in models.items():
        start_time = time.time()
        print("Start to fit model : ", model_name)
        # Fit model
        clf.fit(X_train, y_train)
        print("Model {} fit done".format(model_name))

        y_pred = clf.predict(X_test)

        eval = {"Model": model_name}

        # Evaluate on test data
        metric_names = sorted(list(metrics.keys()))
        for metric_name in metric_names:
            metric_fn = metrics.get(metric_name)
            score_value = metric_fn(y_test, y_pred, LABELS)
            eval.update({metric_name: score_value})

        evaluate_result.append(eval)
        exec_time = time.time() - start_time
        print("Model {} fit and evaluate done. Time : {} seconds".format(model_name, exec_time))

    print("Head of avaluate result")
    evaluate_result = pd.DataFrame(evaluate_result)
    print(evaluate_result.head())

    # Save evaluate result
    save_dir = "./Model/Stacking"
    utils.mkdirs(save_dir)
    save_path = os.path.join(save_dir, "evaluate.csv")

    evaluate_result.to_csv(save_path, index=False)
    print("Save evaluate result to {} done".format(save_path))
