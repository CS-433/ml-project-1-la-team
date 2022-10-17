# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(path_dataset, sub_sample=False):
    """

    """

    # With 2 IO access
    data = np.genfromtxt(path_dataset, delimiter=',', encoding='utf-8',
            skip_header=1, max_rows=50 if sub_sample else None)
    x =  data[:, 2:].copy()

    # Load only the first column to generate the first array
    y_raw = np.genfromtxt(path_dataset, delimiter=',', encoding='utf-8',
            skip_header=1, max_rows=50 if sub_sample else None, 
            usecols=[1], dtype=np.string_)

    y = []
    for sample in y_raw:
        y.append(0 if sample==bytes('b','ascii') else 1)
    y = np.array(y)[:, np.newaxis]
    
    return x, y

    # With 1 IO access
    # data = np.genfromtxt(path_dataset, delimiter=',', encoding='utf-8',
    #           skip_header=0, names = True, dtype = None,
    #           max_rows=50 if sub_sample else None)

    # N, D = len(data), len(data[0])-2
    # x = np.zeros((N, D))
    # y = np.zeros(N)

    # for row_idx, sample in enumerate(data.flat):
    #     x[row_idx] = np.array(sample.tolist()[2:], dtype=float)
    #     y[row_idx] = 0 if sample[1]=='b' else 1
    
    # return x, y


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
    """

    """
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

#Replace each degree feature by a feature of it's sine and one of it's cosine
def expand_degree(x,ID_feature):
    x=np.c_[x,np.cos(x[:,ID_feature])]
    x[:,ID_feature] = np.sin(x[:,ID_feature])
    return x

#For all degree features
def expand_degrees(x,IDs_Features):
    for ids in IDs_Features :
        x=expand_degree(x,ids)
    return x

def build_poly(x, degree):
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly