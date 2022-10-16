# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


# def load_data(path_dataset, sub_sample=True, add_outlier=False):
#     """Load data and convert it to the metric system."""
#     data = np.genfromtxt(path_dataset,
#         skip_header=1, missing_values=['s', 'b'], filling_values=['0', '1'], 
#         max_rows=50 if sub_sample else None)

#     y = ...
    
#     return x, y

def load_data(path_dataset, sub_sample=False, add_outlier=False):
    # Load all data for the x array
    data = np.genfromtxt(path_dataset, delimiter=',', skip_header=1,
            max_rows=50 if sub_sample else None)
    x =  data[:, 2:].copy()

    # Load only the first column to generate the first array
    y_raw = np.genfromtxt(path_dataset, delimiter=',', skip_header=1, 
            max_rows=50 if sub_sample else None, usecols=[1], 
            dtype=np.string_)

    y = []
    for sample in y_raw:
        y.append(0 if sample==bytes('b','ascii') else 1)
    y = np.array(y)[:, np.newaxis]
    
    return x, y


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


def replace_nan_by_means(dataset, nan_value=-999.0, mean_dataset=None):
    """

        mean_dataset : use it if the value has already been computed
            array of shape (D) (contains the mean of each feature column)

        Test :  arr_test = np.array([[1, 2, 3, 4], [10, np.nan, 11, 12], [np.nan, 13, 14, np.nan], [np.nan, 15, 16, 17]])
                arr_test_theoric = replace_nan_by_means(arr_test)
                assert(np.allclose(arr_test_theoric[1, 1], np.nanmean(arr_test[:, 1]))) #, "mean not computed correctly"
    """

    for col_idx in range(dataset.shape[1]):
        dataset_col = dataset[:, col_idx]
        dataset_col[np.isnan(dataset_col)] = mean_dataset[col_idx]

    return dataset

    # def replace_nan_by_feature_mean(feature):
    #     """
    #         input : a columns of the dataset
            
    #     """
    #     feature[np.isnan(feature)] = np.nanmean(feature)
    #     return feature


    # if fill_values is None:
    #     return np.apply_along_axis(replace_nan_by_feature_mean, 0, dataset)
    # else:
    #     dataset[dataset == nan_value] = fill_values
    #     return dataset