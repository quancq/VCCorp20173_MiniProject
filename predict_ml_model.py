import numpy as np
import pandas as pd
import utils, os, json
from ml_model import EnsembleModel, VOCAB_PATH, SCORING
from hyper_parameters import MAP_LABEL_ID_TO_TEXT


def save_predicted_data(df, save_dir):
    predicted_data = []
    for index, row in df.iterrows():
        predicted_data.append({
            "id": row["id"],
            "label": row["LabelID_Pred"],
            "content": row["content"]
        })

    # Save data to file
    utils.mkdirs(save_dir)
    save_path = os.path.join(save_dir, "predicted_data_sent.json")
    with open(save_path, 'w') as f:
        json.dump(predicted_data, f)
    print("Save predicted data to {} done".format(save_path))


if __name__ == "__main__":
    # Load test data
    test_data_path = "./Dataset/data_test.json"
    test_data = utils.load_data(test_data_path)

    df = pd.DataFrame(test_data)
    print(df.head())
    X_test = df.content.values

    # Load model
    model = EnsembleModel(SCORING, VOCAB_PATH)
    # model_dir = "./Model/2018-08-13_01-56-01"
    model_dir = "./Model/2018-08-14_23-08-53"

    model.load_model(model_dir)

    # Predict
    y_pred, prob_pred = model.predict(X_test)
    df["LabelID_Pred"] = y_pred
    df["Label_Pred"] = [MAP_LABEL_ID_TO_TEXT.get(label_id, "Unknown") for label_id in y_pred]
    df["Prob_Pred"] = prob_pred

    print("\nPredict result : ")
    print(df.head())

    # Save predicted data
    # Cach 1
    save_path = os.path.join(model_dir, "predicted_data_{}_{}.csv".format(df.shape[0],utils.get_format_time_now()))
    df.to_csv(save_path, index=False)
    print("Save predicted data to {} done".format(save_path))

    # Cach 2
    save_dir = "./Dataset"
    save_predicted_data(df, save_dir)


