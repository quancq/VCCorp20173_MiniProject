import numpy as np
import pandas as pd
import utils, os, time, math
from preprocessing import FeatureTransformer
from scipy import sparse
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
from hyper_parameters import RANDOM_STATE


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


def get_over_sampling_ratio(labels):
    unique_label = np.unique(labels)
    mean_sample_of_each_label = int(labels.shape[0] / len(unique_label)) + 1
    ratio = {}
    for label in unique_label:
        num_sample_of_label = np.sum(labels == label)
        if num_sample_of_label < mean_sample_of_each_label:
            desired_num_sample = mean_sample_of_each_label - \
                                 int(0.3 * (mean_sample_of_each_label - num_sample_of_label))
            ratio.update({label: desired_num_sample})

    return ratio


def get_under_sampling_ratio(labels):
    unique_label = np.unique(labels)
    mean_sample_of_each_label = int(labels.shape[0] / len(unique_label)) + 1
    ratio = {}
    for label in unique_label:
        num_sample_of_label = np.sum(labels == label)
        if num_sample_of_label > mean_sample_of_each_label:
            desired_num_sample = mean_sample_of_each_label + \
                                 int(0.2 * (num_sample_of_label - mean_sample_of_each_label))
            ratio.update({label: desired_num_sample})

    return ratio


if __name__ == "__main__":
    # Load trining data
    training_encoded_data_path = "./Dataset/encoded_training_data_4362.json"
    X_train, y_train = FeatureTransformer.load_encoded_data(training_encoded_data_path)

    unique_label = np.unique(y_train)
    print("Num distinct labels : ", len(unique_label))
    mean_sample_of_each_label = int(X_train.shape[0] / len(unique_label))
    print("Mean sample of each label : ", mean_sample_of_each_label)
    # ratio = {label: mean_sample_of_each_label for label in unique_label}
    # ratio = get_over_sampling_ratio(y_train)
    # print("Ratio size : ", len(ratio))
    # print(ratio)

    # Resampling
    # new_X_train, new_y_train = over_sampling(X_train, y_train)
    # Over sampling
    over_ratio = get_over_sampling_ratio(y_train)
    smt = SMOTE(random_state=RANDOM_STATE, ratio=over_ratio, k=4)
    new_X_train, new_y_train = smt.fit_sample(X_train, y_train)

    # Under sampling
    under_ratio = get_under_sampling_ratio(new_y_train)
    cc = ClusterCentroids(random_state=RANDOM_STATE, ratio=under_ratio)
    new_X_train, new_y_train = cc.fit_sample(new_X_train, new_y_train)

    # Save new data
    training_new_encoded_data_path = "./Dataset/encoded_smote-cc_training_data_{}.json".format(new_y_train.shape[0])
    FeatureTransformer.save_encoded_data(new_X_train, new_y_train, training_new_encoded_data_path)
