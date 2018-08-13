import numpy as np
import pandas as pd
import utils, os
from ml_model import EnsembleModel, VOCAB_PATH, SCORING
from hyper_parameters import MAP_LABEL_ID_TO_TEXT
from sklearn.metrics import get_scorer, make_scorer, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


if __name__ == "__main__":
    # Load test data
    test_data_path = "./Dataset/valid_data_1091.json"
    test_data = utils.load_data(test_data_path)[:300]

    df = pd.DataFrame(test_data)
    print("Head of test data:")
    print(df.head())
    X_test = df.content.values
    y_test = df.label.values

    # Load model
    model = EnsembleModel(SCORING, VOCAB_PATH)
    # model_dir = "./Model/2018-08-13_01-56-01"
    model_dir = "./Model/2018-08-13_09-25-33"
    model.load_model(model_dir)

    # Evaluate
    # metrics = ["f1_macro", "precision_macro", "recall_macro", "accuracy"]
    metrics = {
        "f1_macro": {"fn": f1_score, "params": {"average": "macro"}},
        "accuracy": {"fn": accuracy_score}
    }
    result_df, cf_mats = model.evaluate(X_test, y_test, metrics)

    print("\nEvaluate result:")
    print(result_df)

    # Save confusion matrix figure
    save_dir = os.path.join(model_dir, "Evaluate")
    utils.plot_multi_confusion_matrix(cf_mats, save_dir)
