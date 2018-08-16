# File contains all hyper parameters

MAX_WORD_LENGTH = 20
MIN_WORD_LENGTH = 2
NUM_LABELS = 22
MIN_OCCURRENCES_TOKEN = 3

SCORING = ["f1_macro", "f1_micro", "accuracy"]
CV = 2
RANDOM_STATE = 7
VOCAB_PATH = "./Vocabulary/vocab_17012.csv"

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

LABELS = sorted(list(MAP_LABEL_ID_TO_TEXT.keys()))
