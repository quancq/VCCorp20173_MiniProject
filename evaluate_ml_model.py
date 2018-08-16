import numpy as np
import pandas as pd
import utils, os
from ml_model import EnsembleModel, VOCAB_PATH, SCORING
from hyper_parameters import MAP_LABEL_ID_TO_TEXT
from sklearn.metrics import get_scorer, make_scorer, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


if __name__ == "__main__":
    test_data_path = "./Dataset/data_test.json"
    model_dir = "./Model/2018-08-14_23-08-53"

    # Load test data
    test_data = utils.load_data(test_data_path)

    df = pd.DataFrame(test_data)
    print("Head of test data:")
    print(df.head())
    X_test = df.content.values
    y_test = df.label.values
    unique_labels = np.unique(y_test)

    # Load model
    model = EnsembleModel(SCORING, VOCAB_PATH)
    model.load_model(model_dir)

    # Evaluate
    metrics = {
        "f1_macro": {"fn": f1_score, "params": {"average": "macro"}},
        # "f1_micro": {"fn": f1_score, "params": {"average": "micro"}},
        "accuracy": {"fn": accuracy_score},
        "precision_macro": {"fn": precision_score, "params": {"average": "macro"}},
        # "precision_micro": {"fn": precision_score, "params": {"average": "micro"}},
        "recall_macro": {"fn": recall_score, "params": {"average": "macro"}},
        # "recall_micro": {"fn": recall_score, "params": {"average": "micro"}}
    }
    is_predict_proba = False
    result_df, cf_mats = model.evaluate(X_test, y_test, metrics, is_predict_proba=is_predict_proba)

    print("\nEvaluate result:")
    print(result_df)

    # Save confusion matrix figure
    eval_save_dir = os.path.join(model_dir, "Evaluate_Proba" if is_predict_proba else "Evaluate")
    cm_save_dir = os.path.join(eval_save_dir, "Confusion_Matrix")
    utils.plot_multi_confusion_matrix(cf_mats, cm_save_dir)

    # Save result plot
    utils.plot_multi_bar_with_annot(result_df, fig_save_dir=eval_save_dir, is_plot=False)

    result_save_path = os.path.join(eval_save_dir, "evaluate_{}.csv".format(len(y_test)))
    result_df.to_csv(result_save_path, index=False)
    print("Save file evaluate to {} done".format(result_save_path))
