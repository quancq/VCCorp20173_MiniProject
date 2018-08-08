import os, time, json, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split


def load_data(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    print("Read file {} done".format(file_path))
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


def plot_stats_count(data):
    # Plot bar chart with x-axis is column 1, y-axis is column 2

    ax = data.plot(kind="bar", x=0, y=1, legend=False, figsize=(12, 6), rot=0)
    col_names = list(data.columns.values)
    xlabel = col_names[0]
    ylabel = col_names[1]
    ax.set(xlabel=xlabel, ylabel=ylabel)

    mean = data.iloc[:, 1].mean()
    ax.axhline(y=mean, linestyle='--', color='black', linewidth=1)
    ax.set_yticks(list(ax.get_yticks()) + [mean])

    x_offset = -0.3
    y_offset = 5

    # add value into plot to see clearly
    for p in ax.patches:
        b = p.get_bbox()
        value = int(b.y1)
        ax.annotate(value, xy=((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))

    fig_path = "./ExploreResult/{}-{}.png".format(xlabel, ylabel)
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
    mpl.rcParams['font.size']=12                #10
    mpl.rcParams['savefig.dpi']=100             #72
    mpl.rcParams['figure.subplot.bottom']=.1

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


def split_data(data, test_size=0.2):
    X, y = convert_orginal_data_to_list(data)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, stratify=y, random_state=7)
    training_data = convert_two_list_to_dicts(X_train, y_train)
    valid_data = convert_two_list_to_dicts(X_valid, y_valid)

    return training_data, valid_data


if __name__ == "__main__":
    training_file_path = "./Dataset/data_train.json"
    test_file_path = "./Dataset/data_sent.json"

    training_data = load_data(training_file_path)
    training_size = len(training_data)
    test_data = load_data(test_file_path)
    test_size = len(test_data)

    # training_data = convert_original_data_to_dict(training_data)
    training_data, valid_data = split_data(training_data)

    # Save splitted data
    training_path = "./Dataset/training_data_{}.json".format(len(training_data))
    with open(training_path, 'w', encoding="utf-8") as f:
        json.dump(training_data, f)

    valid_path = "./Dataset/valid_data_{}.json".format(len(valid_data))
    with open(valid_path, 'w', encoding="utf-8") as f:
        json.dump(valid_data, f)
