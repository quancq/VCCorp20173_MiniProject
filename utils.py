import os, time, json, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
import itertools
from datetime import datetime
from hyper_parameters import LABELS


def upper(s):

    if isinstance(s, type(u"")):
        return s.upper()
    return str(s, "utf8").upper().encode("utf8")


def lower(s):
    if isinstance(s, type(u"")):
        return s.lower()
    return str(s, "utf8").lower().encode("utf8")


def load_data(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    print("Read file {} done. Data size : {}".format(file_path, len(data)))
    return data


def convert_original_data_to_df(data):
    dic = {}

    for col in data[0].keys():
        values = [elm.get(col) for elm in data]
        dic.update({col: values})
    df = pd.DataFrame(dic)

    return df


def convert_original_data_to_dict(data):
    dic = {}
    for doc in data:
        label = doc.get("label")
        content = doc.get("content")
        doc_list = dic.get(label)
        if doc_list is None:
            doc_list = []
            dic.update({label: doc_list})

        doc_list.append(content)

    return dic


def plot_stats_count(data, is_save=False):
    # Plot bar chart with x-axis is column 1, y-axis is column 2
    mpl.style.use("seaborn")

    col_names = list(data.columns.values)
    xlabel = col_names[0]
    ylabel = col_names[1]
    data_size = data[ylabel].sum()

    data[ylabel] = data[ylabel] * 100 / data[ylabel].sum()
    ax = data.plot(kind="bar", x=0, y=1, legend=False, figsize=(12, 6), rot=0, color="C1")
    ax.set(xlabel=xlabel, ylabel=ylabel)

    mean = data.iloc[:, 1].mean()
    ax.axhline(y=mean, linestyle='--', color='black', linewidth=1)
    ax.set_yticks(list(ax.get_yticks()) + [mean])

    x_offset = -0.3
    y_offset = 0.3

    # add value into plot to see clearly
    for p in ax.patches:
        b = p.get_bbox()
        value = "{:.2f}".format((b.y1))
        ax.annotate(value, xy=((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

    if is_save:

        fig_path = "./ExploreResult/{}-{}_{}.png".format(xlabel, ylabel, data_size)
        plt.savefig(fig_path, dpi=300)
        print("Save figure to ", fig_path)
    plt.show()


def write_sample_dataset(data, num_samples=5):
    dir_path = "../SampleDataset"

    for label, docs in data.items():
        file_path = os.path.join(dir_path, "{}.txt".format(label))
        with open(file_path, 'w', encoding="utf-8") as f:
            for i, doc in enumerate(docs[:num_samples]):
                f.write("[Article {}]\n{}\n\n".format(i + 1, doc))


def plot_word_cloud(corpus, file_path):
    # mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
    mpl.rcParams['font.size'] = 12  # 10
    mpl.rcParams['savefig.dpi'] = 100  # 72
    mpl.rcParams['figure.subplot.bottom'] = .1

    stopwords = set(STOPWORDS)
    print("Stopword size : ", len(stopwords))

    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        # max_words=200,
        max_font_size=40,
        random_state=42
    ).generate(corpus)

    print(wordcloud)
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    # plt.show()
    plt.savefig(file_path, dpi=900)


def convert_orginal_data_to_list(data):
    X = []
    y = []

    for doc in data:
        X.append(doc.get("content"))
        y.append(doc.get("label"))

    return X, y


def convert_two_list_to_dicts(contents, labels):
    lst = []
    for content, label in zip(contents, labels):
        lst.append({
            "label": label,
            "content": content
        })

    return lst


# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          save_path,
                          normalize=True,
                          is_plot=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float')
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm, cm_sum, out=np.zeros_like(cm), where=cm_sum != 0)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print("Confusion matrix shape : ", cm.shape)
    # print(cm)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")
    # plt.colorbar()
    #
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes)
    # plt.yticks(tick_marks, classes)
    #
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     s = format(cm[i, j], fmt) if cm[i, j] > 0 else 0
    #     plt.text(j, i, s,
    #              ha="center", va="center",
    #              color="white" if cm[i, j] > thresh else "black")
    #
    # plt.tight_layout()

    # _, ax = plt.subplots()
    # mask = (cm == 0)
    cm_df = pd.DataFrame(cm)
    cm_df = cm_df.applymap(lambda x: "{:g}".format(round(x, 2)) if x > 0 else "0")
    # print(cm_df.head())

    plt.figure(figsize=(12, 12))

    sns.heatmap(cm, cmap=cmap, annot=cm_df, xticklabels=classes, yticklabels=classes, fmt='')
    plt.title(title, fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)
    plt.ylabel("True label", fontsize=20)
    plt.yticks(rotation=0)

    # Save figure
    plt.savefig(save_path)

    if is_plot:
        plt.show()


def plot_multi_confusion_matrix(cf_mats, save_dir):
    mpl.style.use("seaborn")
    print("Start to plot multi confusion matrix to ", save_dir)
    mkdirs(save_dir)

    for model_name, (cf_mat, unique_label) in cf_mats.items():
        save_path = os.path.join(save_dir, "{}.png".format(model_name))
        plot_confusion_matrix(cf_mat, unique_label, save_path)

    print("Plot {} confusion matrix to {} done".format(len(cf_mats), save_dir))


def plot_bar_with_annot(x, y, xlabel, ylabel, title="", fig_save_dir=None, is_plot=True):
    mkdirs(fig_save_dir)
    x_offset = -0.03
    y_offset = 0.01
    mpl.style.use("seaborn")

    # Sort by ascending score
    arg_sorted = np.argsort(y)
    x = x[arg_sorted]
    y = y[arg_sorted]

    fig, ax = plt.subplots()
    ax.bar(x=x, height=y, color='C1', width=0.25)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.tick_params(rotation=0)

    # Set lower and upper limit of y-axis
    min_score = y.min()
    max_score = y.max()
    y_lim_min = (min_score - 0.2) if min_score > 0.2 else 0
    y_lim_max = (max_score + 1) if max_score > 1 else 1
    ax.set_ylim([y_lim_min, y_lim_max])

    # Show value of each column to see clearly
    for p in ax.patches:
        b = p.get_bbox()
        text_value = "{:.4f}".format(b.y1)
        ax.annotate(text_value, xy=(b.x0 + x_offset, b.y1 + y_offset))

    if fig_save_dir is not None:
        fig_save_path = os.path.join(fig_save_dir, "{}.png".format(ylabel))
        plt.savefig(fig_save_path)
        print("Plot and save figure to {} done".format(fig_save_path))
    if is_plot:
        plt.show()


def plot_multi_bar_with_annot(data_plot, fig_save_dir, is_plot=True):
    mkdirs(fig_save_dir)
    columns = list(data_plot.columns)
    print("Start to plot and save {} figures to {} ...".format(len(columns) - 1, fig_save_dir))

    print("Head of data plot")
    print(data_plot.head())
    mpl.style.use("seaborn")

    xlabel = columns[0]
    for ylabel in columns[1:]:
        x = data_plot[xlabel].values
        y = data_plot[ylabel].values
        plot_bar_with_annot(x, y, xlabel, ylabel, title="{}-{}".format(ylabel, xlabel),
                            fig_save_dir=fig_save_dir, is_plot=is_plot)

    print("Plot {} figures done".format(len(columns) - 1))


def split_data(data, test_size=0.2):
    X, y = convert_orginal_data_to_list(data)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, stratify=y, random_state=7)
    training_data = convert_two_list_to_dicts(X_train, y_train)
    valid_data = convert_two_list_to_dicts(X_valid, y_valid)

    return training_data, valid_data


def write_vocab(vocab, save_path):
    words = list(vocab.keys())
    words.sort()

    with open(save_path, 'w', encoding="utf-8") as f:
        f.write("\n".join(words))

    print("Write vocab (size = {}) to {} done".format(len(words), save_path))


def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        print("Create new directory : ", dir_path)
        os.makedirs(dir_path)


def get_format_time_now():
    dt = datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def filter_data_by_attrib(data, attrib, remove_ids):
    print("Before remove {} ({}), data size = {}".format(attrib, remove_ids, len(data)))
    filtered_data = []
    for data_point in data:
        if data_point[attrib] not in remove_ids:
            filtered_data.append(data_point)

    print("After remove {} ({}), data size = {}".format(attrib, remove_ids, len(filtered_data)))
    return filtered_data


if __name__ == "__main__":
    # training_file_path = "./Dataset/data_train.json"
    # test_file_path = "./Dataset/data_sent.json"
    #
    # training_data = load_data(training_file_path)
    # training_size = len(training_data)
    # test_data = load_data(test_file_path)
    # test_size = len(test_data)

    words = ["Và", "CÓ", "ĐưỢc", "KhÔnG", "Ộ uẾ qUá NặnG", "Ả á à Ê"]
    for word in words:
        print("Original word : {} - Lower : {} - Upper : {}".format(word, lower(word), upper(word)))
