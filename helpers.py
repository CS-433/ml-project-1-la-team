# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from implementations import sigmoid
import csv

def remove_constant_features(tx):
    constant_ind = np.where(np.nanstd(tx, axis=0) == 0)[0] #std = 0
    return np.delete(tx, constant_ind , axis=1)

def get_split_by_jet_data(y,tx,jet):
    indices = np.where(tx[:, 18] == jet)
    return y[indices], tx[indices]

def robust_scaling(tx,q1,q2,q3): 
    """Robust scaling """
    res = (tx - q2) / (q3 - q1)
    res[:,17] = tx[:,17]
    return res

def remove_outliers(tx,q1,q3):
    """
    Use IQR method from https://online.stat.psu.edu/stat200/lesson/3/3.2
    """
    iqr = q3 - q1
    outq1 = np.where(tx < q1 - 1.5 * iqr)
    print("Outliers < q1 : " + str(outq1) +","+  str(len(outq1[0])))
    outq3 = np.where(tx > q3 + 1.5 * iqr)
    print("Outliers > q3 : " + str(outq1) +","+  str(len(outq1[0])))
    tx[outq1] = np.take(q1 - 1.5 * iqr,outq1[1])
    tx[outq3] = np.take(q3 + 1.5 * iqr,outq3[1])
    return tx

def transform(tx,IDs_degrees):
    """Remove constants,handle degrees, remove outliers, standardize with robust scaling"""
    tx = remove_constant_features(tx)
    tx = expand_degrees(tx,IDs_degrees)
    q1 = np.nanpercentile(tx,q=25,axis = 0)
    q2 = np.nanpercentile(tx,q=50,axis = 0)
    q3 = np.nanpercentile(tx,q=75,axis = 0)

    tx = remove_outliers(tx,q1,q3)

    return robust_scaling(tx,q1,q2,q3)

def standardize_training(x, missing_values=True):
    """Standardize the original data set."""
    mean_x = np.nanmean(x, axis=0) if missing_values else np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.nanstd(x, axis=0) if missing_values else np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def standardize_test(x, mean_x, std_x):
    """"""
    return (x - mean_x) / std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def add_offset(x):
    """ """
    return np.hstack((np.ones(x.shape[0])[:, np.newaxis], x))


def replace_nan_by_means(data, nan_value=-999.0, mean_data=None):
    """

    mean_dataset : use it if the value has already been computed
        array of shape (D) (contains the mean of each feature column)

    Test :  arr_test = np.array([[1, 2, 3, 4], [10, np.nan, 11, 12], [np.nan, 13, 14, np.nan], [np.nan, 15, 16, 17]])
            arr_test_theoric = replace_nan_by_means(arr_test)
            assert(np.allclose(arr_test_theoric[1, 1], np.nanmean(arr_test[:, 1]))) #, "mean not computed correctly"
    """

    for col_idx in range(data.shape[1]):
        dataset_col = data[:, col_idx]
        dataset_col[np.isnan(dataset_col)] = mean_data[col_idx]

    return data


# Replace each degree feature by a feature of it's sine and one of it's cosine
def expand_degree(x, ID_feature):
    x = np.c_[x, np.cos(x[:, ID_feature])]
    x[:, ID_feature] = np.sin(x[:, ID_feature])
    return x


# For all degree features
def expand_degrees(x, IDs_Features):
    for ids in IDs_Features:
        x = expand_degree(x, ids)
    return x

def build_poly(x, degree):
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def get_col_idx(col_name, col_names):
    """
        Get col index given the feature name (for the initial features only)
    """
    return [col_idx-2 for col_idx, name in enumerate(col_names) if col_name == name][0]

def predict(w,x):
    y_predict = sigmoid(x @ w)

    y_predict[np.where(y_predict <= 0.5)] = -1
    y_predict[np.where(y_predict > 0.5)] = 1

    return y_predict