import numpy as np
import pandas as pd
import utils, os
from ml_model import EnsembleModel, VOCAB_PATH, SCORING

MAP_LABEL_ID_TO_TEXT = {
    1: "Thông số kĩ thuật",
    2: "Viễn thông",
    6: "Du lịch",
    7: "Giáo dục",
    8: "Sản phẩm",
    9: "Công thức nấu ăn",
    10: "Y tế",
    11: "Mỹ phẩm",
    12: "Thời trang",
    13: "Bất động sản",
    14: "Thiết kế nội thất",
    15: "Ẩm thực",
    16: "Xe cộ",
    17: "Đời sống - Chứng khoán",
    19: "Kinh tế - Chính trị - Xã hội",
    20: "Nông nghiệp - Công nghiệp",
    21: "Quảng cáo công ty",
    22: "Kế toán - Kiểm toán",
    23: "Game",
    24: "Mẹ và bé",
    156: "Hàng không",
    188: "Bóng đá",
}

if __name__ == "__main__":
    # Load test data
    test_data_path = "./Dataset/data_sent.json"
    test_data = utils.load_data(test_data_path)

    df = pd.DataFrame(test_data)
    print(df.head())
    X_test = df.content.values

    # Load model
    model = EnsembleModel(SCORING, VOCAB_PATH)
    model_dir = "./Model/2018-08-13_01-56-01"
    model.load_model(model_dir)

    y_pred, prob_pred = model.predict(X_test)
    df["LabelID_Pred"] = y_pred
    df["Label_Pred"] = [MAP_LABEL_ID_TO_TEXT.get(label_id, "Unknown") for label_id in y_pred]
    df["Prob_Pred"] = prob_pred

    print("\nPredict result : ")
    print(df.head())

    # Save predicted data
    save_path = os.path.join(model_dir, "predicted_data_{}_{}.csv".format(df.shape[0],utils.get_format_time_now()))
    df.to_csv(save_path, index=False)
