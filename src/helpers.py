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


def standardize(x, missing_values=False):
    """Standardize the original data set."""
    mean_x = np.nanmean(x) if missing_values else np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


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

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
        
    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    N = x.shape[0]
    tr_size = int(N*ratio)
    perms = np.random.permutation(N)

    return x[perms[:tr_size]], x[perms[tr_size:]], y[perms[:tr_size]], y[perms[tr_size:]]
