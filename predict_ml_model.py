import numpy as np
import pandas as pd
import utils, os
from ml_model import EnsembleModel, VOCAB_PATH, SCORING

if __name__ == "__main__":
    # Load test data
    test_data_path = "./Dataset/data_sent.json"
    test_data = utils.load_data(test_data_path)

    df = pd.DataFrame(test_data)
    print(df.head())
    X_test = df.content.values

    # Load model

    model = EnsembleModel(SCORING, VOCAB_PATH)
    model_dir = "./Model/2018-08-11_19-21-22"
    model.load_model(model_dir)

    y_pred, prob_pred = model.predict(X_test)
    df["Label_Pred"] = y_pred
    df["Prob_Pred"] = prob_pred

    print("\nPredict result : ")
    print(df.head())

    # Save predicted data
    save_path = os.path.join(model_dir, "predicted_data.csv")
    df.to_csv(save_path, index=False)
