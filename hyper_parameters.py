# File contains all hyper parameters

MAX_WORD_LENGTH = 20
MIN_WORD_LENGTH = 2
NUM_LABELS = 22
MIN_OCCURRENCES_TOKEN = 3

SCORING = ["f1_macro", "accuracy"]
CV = 2
RANDOM_STATE = 7
VOCAB_PATH = "./Vocabulary/vocab_17990.csv"

MAP_LABEL_ID_TO_TEXT = {
    1: "Điện tử - Điện lạnh",
    2: "Viễn thông - Truyền thông",
    6: "Du lịch - Vận tải",
    7: "Việc làm - Giáo dục",
    8: "Hàng tiêu dùng",
    9: "Thực phẩm - Đồ uống",
    10: "Y tế - Thuốc",
    11: "Chăm sóc sắc đẹp",
    12: "Thời trang - Trang sức",
    13: "Xây dựng - Bất động sản",
    14: "Đồ nội thất, ngoại thất",
    15: "Nhà hàng, quán bar, trung tâm giải trí",
    16: "Ô tô - Xe máy",
    17: "Tài chính, bảo hiểm",
    18: "Bán buôn, bán lẻ",
    19: "Luật - Chính phủ",
    20: "Ngành công nghiệp, đặc thù",
    21: "Dịch vụ",
    22: "Kế toán - Kiểm toán",
    23: "Game",
    24: "Mẹ và bé",
    156: "Hàng không",
    188: "Thể thao",
}

LABELS = sorted(list(MAP_LABEL_ID_TO_TEXT.keys()))
REMOVE_LABEL_IDS = [18]
# REMOVE_DOC_IDS = [1,3,7,8,9,10,11,12,95,98,99,108,143,144,145,148,149,150,151,152,183,187,266,286,294,335]
REMOVE_DOC_IDS = [9, 143, 145, 149, 150, 152, 153, 155, 156, 160]
