import numpy as np
import pandas as pd
import utils, os, time, math
from preprocessing import FeatureTransformer
from scipy import sparse


def over_sampling_one_label(original_data, sampling_size):
    print("Sampling one label: original shape = {}, sampling size = {}".format(original_data.shape, sampling_size))
    original_size = original_data.shape[0]
    num_replication = sampling_size - original_size
    if num_replication <= 0:
        return original_data

    padded_data = []
    index = 0
    for i in range(num_replication):
        padded_data.append(original_data[index].copy())
        index = (index + 1) % original_size

    padded_data = sparse.vstack(padded_data)
    new_data = sparse.vstack((original_data, padded_data))

    print("After sampling : new data shape = ", new_data.shape)
    return new_data


def over_sampling(X, y):
    print("Before over sampling: Xshape = {}, Yshape = {}".format(X.shape, y.shape))
    unique_labels = np.unique(y)
    sampling_size = int(math.ceil(X.shape[0] / unique_labels.shape[0]))
    new_data = []
    new_label = []
    for label in unique_labels:
        idx = np.where(y == label)[0]
        data = X[idx]
        new_data_of_label = over_sampling_one_label(data, sampling_size)
        new_data.append(new_data_of_label)
        new_label.extend([label for _ in range(new_data_of_label.shape[0])])

    new_data = sparse.vstack(new_data)
    new_label = np.array(new_label)

    print("After over sampling: Xshape = {}, Yshape = {}".format(new_data.shape, new_label.shape))
    return new_data, new_label


if __name__ == "__main__":
    # Load trining data
    training_encoded_data_path = "./Dataset/encoded_training_data_4362.json"
    X_train, y_train = FeatureTransformer.load_encoded_data(training_encoded_data_path)

    # Over sampling
    new_X_train, new_y_train = over_sampling(X_train, y_train)

    # Save new data
    training_new_encoded_data_path = "./Dataset/encoded_over_training_data_{}.json".format(new_y_train.shape[0])
    FeatureTransformer.save_encoded_data(new_X_train, new_y_train, training_new_encoded_data_path)
