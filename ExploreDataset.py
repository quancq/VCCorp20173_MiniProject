import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


if __name__ == "__main__":
    # Load data to explore
    training_file_path = "./Dataset/data_train.json"
    test_file_path = "./Dataset/data_sent.json"

    training_data = utils.load_data(training_file_path)
    training_size = len(training_data)
    test_data = utils.load_data(test_file_path)
    test_size = len(test_data)

    print("Training data size : ", training_size)
    print("Test data size : ", test_size)

    print("========================================")

    training_df = utils.convert_dicts_to_df(training_data)

    print(training_df.info())

    print("\nStatistic")
    stats_by_label = training_df.label.value_counts().sort_index().reset_index()
    cols = ["label", "total"]
    stats_by_label.columns = cols
    print(stats_by_label)
    print("Number distinct label : ", stats_by_label.shape[0])

    utils.plot_stats_count(stats_by_label)

