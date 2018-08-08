import os, time, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    print("Read file {} done".format(file_path))
    return data


def convert_dicts_to_df(data):
    dic = {}

    for col in data[0].keys():
        values = [elm.get(col) for elm in data]
        dic.update({col: values})
    df = pd.DataFrame(dic)

    return df


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
    plt.savefig(fig_path)
    print("Save figure to ", fig_path)
    plt.show()
