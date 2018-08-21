import numpy as np
import pandas as pd
import utils, os
from ml_model import EnsembleModel, VOCAB_PATH, SCORING
from hyper_parameters import MAP_LABEL_ID_TO_TEXT, REMOVE_DOC_IDS, REMOVE_LABEL_IDS
from sklearn.metrics import get_scorer, make_scorer, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


if __name__ == "__main__":
    test_data_path = "./Dataset/data_test_774.json"
    model_dir = "./Model/2018-08-20_23-57-43"

    # Load test data
    test_data = utils.load_data(test_data_path)

    # Filter test data
    # test_data = utils.filter_data_by_attrib(test_data, "id", REMOVE_DOC_IDS)
    # test_data = utils.filter_data_by_attrib(test_data, "label", REMOVE_LABEL_IDS)

    df = pd.DataFrame(test_data)
    print("Head of test data:")
    print(df.head())
    X_test = df.content.values
    y_test = df.label.values
    unique_labels = np.unique(y_test)

    # Load model
    model = EnsembleModel(SCORING, VOCAB_PATH)
    model.load_model(model_dir)
    # model.remove_model("KNN")
    # model.remove_model("RandomForest")
    # model.remove_model("ExtraTree")
    # model.remove_model("LightGBM")

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

    # Save file contain y_pred and y_true
    y_pred, prob_pred = model.predict(X_test)
    df["LabelID_Pred"] = y_pred
    df["Label_Pred"] = [MAP_LABEL_ID_TO_TEXT.get(label_id, "Unknown") for label_id in y_pred]
    df["Prob_Pred"] = prob_pred

    save_path = os.path.join(eval_save_dir, "evaluate_data_{}_{}.csv".format(df.shape[0], utils.get_format_time_now()))
    df.to_csv(save_path, index=False)
    print("Save file compare y_true and y_pred to {} done".format(save_path))
