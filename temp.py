import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import STOP_WORDS
from preprocessing import FeatureTransformer
from pyvi import ViTokenizer


if __name__ == "__main__":
    doc = "Bạn là một người mẹ nhất mực thương con, và luôn mong muốn đem đến cho con những điều tốt đẹp nhất từ thể chất đến tinh thần cho con. Và tất nhiên một nguồn dinh dưỡng đảm bảo cho sự phát triển sẽ không thể nào mà thiếu được. Nhưng đôi khi mẹ lại không biết làm sao, làm như thế nào để có thể cân đo ước lượng đúng chuẩn số lượng nguyên liệu để chế biến lên cho con những món ăn không chỉ thơm ngon mà còn chuẩn khoa học nữa, không thừa mà cũng chẳng thiếu. Hãy cùng Sangtao88 tìm hiểu nhé! Vậy để mẹ dễ dàng hơn trong việc này thì sau đây chúng tôi sẽ cung cấp đến bạn một chiếc cân tiểu ly. Vật dụng “thần thánh” không thể nào mà thiếu vắng được trong những lúc mẹ cân đong lượng thức ăn nấu cho con. Hiểu hơn về công dụng cân tiểu ly trong việc cân đong nguyên liệu dinh dưỡng cho con ăn: Cân tiểu ly có tác dụng điều chỉnh lượng dinh dưỡng phù hợp cho bé Nếu như bạn đang muốn nấu cho bé một nồi súp gà thật ngon và thật nhiều dinh dưỡng nhưng bạn lại không biết nên nêm gia vị cũng như chuẩn bị nguyên vật liệu bao nhiêu cho đủ. Bạn lên mạng và search cách nấu và rồi bạn ước lượng bằng tay và sự hiểu biết cùng với suy nghĩ của mình thế này là đủ. Nhưng đến khi nồi sup hoàn thành thì ngay cả bạn cũng cảm thấy không hài lòng với nồi súp này chứ không phải chỉ riêng bé. Nguyên nhân không hẳn là ở việc bạn không biết công thức nấu cũng không biết nguyên liệu nấu hay do bạn nấu dở mà nguyên nhân chính là ở chỗ. Bạn không cân đo được chuẩn lượng gia vị cũng như là lượng nguyên liệu cần thiết để có thể nấu được nồi súp giống như ngoài nhà hàng dành cho bé con của bạn. Cân tiểu ly cân bằng dinh dưỡng cho trẻ Vì thế mà những lúc như này đây bạn hãy chạy ngay đến Sangtao88 chúng tôi hoặc truy cập vào website sangtao88.vn click vào mục cân tiểu ly và tìm mua ngay cho mình một chiếc cân tiểu ly bỏ túi để những lúc mẹ chế biến thức ăn cho con sẽ được dễ dàng hơn. Bởi vì với người lớn bạn có thể ăn nhiều hơn 1 chút so với lượng thức ăn cần tiêu thụ thì có thể chấp nhận được nhưng với con trẻ thì điều này không nên 1 chút nào. Bởi vì hệ tiêu hóa của trẻ lúc này chưa thực sự hoàn thiện nên nếu như mẹ không cần thận trong việc này thì bé có thể sẽ bị bội thực, khó tiêu và sợ ăn. Ngược lại nếu như nguồn dinh dưỡng cấp cho bé không đủ thì sẽ khó lòng đảm bảo cho sự phát triển lớn mạnh của bé được. Vì thế mà mẹ nên có trong tủ bếp nhà mình một chiếc cân tiểu ly mini nhỏ xinh để dễ dàng hơn trong việc chế biến đồ ăn thức uống cho con nhé. >>> Tham khảo bài viết chi tiết tại: http://sangtao88.vn/su-can-thiet-cua-can-tieu-ly-voi-che-do-dinh-duong-cho-tre-nho"

    doc = "http://baoanjsc.com.vn/Images/SanPham/HinhTo/Khoi%20CPU%20OMRON%20CP1E%2011122014054159.jpg Thông số kỹ thuật Nguồn cấp 100 to 240 VAC 50/60 Hz Số lượng ngõ vào/ra 8/6 Ngõ vào 5VDC, 24VDC Ngõ ra Transistor (sinking) Dung lượng chương trình 8K steps Dung lượng bộ nhớ 8K words Thời gian lưu trữ Khoảng 13.000 giờ (Nhiệt độ môi trường xung quanh: 55 ° C) Pin CP1W-BAT01 (Đặt hàng riêng) Chế độ hoạt động - PROGRAM mode: Tại chế độ này thì các thay đổi có thể thực hiện trước khi chạy chương trình - MONITOR mode: Có thể chỉnh sửa trực tiếp và thay đổi các giá trị I/O đặt trước ở chế độ này - RUN mode: Chương trình được thực hiện (Đây là chế độ hoạt động bình thường) Truyền thông RS 232, USB CÔNG TY CỔPHẦN DỊCH VỤ KỸ THUẬT BẢO AN Web: http://baoanjsc.com.vn/ Email: nguyenut.baoan@gmail.com Mobile:01662. 976 .894 /0936.630.266 Link sản phẩm: Khối CPU Omron CP1E-N14DT-A Công ty Cổ phần Dịch vụ và Kỹ thuật Bảo An hiện là nhà phân phối chính thức của các Tập đoàn Omron, Autonics,Yaskawa, SMC và là đại lý của nhiều tập đoàn điện tự động hóa nổi tiếng kháctrên thế giới. Do đó Bảo An cung cấp thiết bị điện tự động hóa với giá tốt nhất. Bảo An códịch vụ đào tạo lập trình PLC , thi công lắp đặt biến tần và hướng dẫn cài đặt biến tần . Các thiết bị điện tự động hóa thông dụng : cảm biến từ Omron , điềukhiển nhiệt độ, rơ le bán dẫn, bộ đếm, timer, bộ nguồn, công tắc hành trình Omron , encoder, plc, màn hình cảm ứng, servo, cảmbiến quang Omron , cảm biến nhiệt độ Omron , cảm biến áp suất Omron , Khối CPU Omron thiết bị đo lường giám sát năng lượng củaomron..., thiết bị omron chính hãng, thiết bị omron bảo hành 24 tháng. Hỗ trợmiễn phí vận chuyển, tư vấn sửa chữa thiết bị Omron."

    # doc1 = ["gia dinh toi"]
    # doc2 = ["gia toi dinh toi la"]
    # doc3 = ["gia dinh toi toi"]

    # tfidf = TfidfVectorizer(stop_words=STOP_WORDS)
    # tokens = [ViTokenizer.tokenize(doc)]
    # doc = [doc]
    # tfidf.fit(doc, np.zeros_like(doc))

    ft = FeatureTransformer()
    doc = [doc]
    ft.fit(doc, np.zeros_like(doc))

    vocab = ft.get_vocab()
    print(vocab)
    print(len(vocab))
    #
    # X1 = tfidf.transform(doc1)
    # X2 = tfidf.transform(doc2)
    # X3 = tfidf.transform(doc3)
    # print("X1 = ", X1)
    # print("X2 = ", X2)
    # print("X3 = ", X3)
    # print(tokens)
