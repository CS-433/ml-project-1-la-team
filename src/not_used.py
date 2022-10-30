"""
@Author Kilian Raude, Colin Pelletier, Joris Monnet
"""
import numpy as np

#
# Pre Processing
#


def standardize_training(x, missing_values=True):
    """Standardize with mean and std training"""
    mean_x = np.nanmean(x, axis=0) if missing_values else np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.nanstd(x, axis=0) if missing_values else np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def standardize_test(x, mean_x, std_x):
    """Standardize test with mean and std"""
    return (x - mean_x) / std_x


def add_offset(x):
    """Add a column of "1" as offset"""
    return np.hstack((np.ones(x.shape[0])[:, np.newaxis], x))


def balance_dataset(x_tr, y_tr):
    """Balance the dataset between s and b labels"""
    nb_s = len(y_tr[y_tr == 1])
    nb_b = len(y_tr) - nb_s

    # select ids from both classes
    idx_s = np.where(y_tr == 1)[0]
    idx_b = np.where(y_tr == 0)[0]

    idx_to_select = np.random.permutation(nb_b)[:nb_s]

    x_tr_s, y_tr_s = x_tr[idx_s], y_tr[idx_s]
    x_tr_b, y_tr_b = x_tr[idx_b][idx_to_select], y_tr[idx_b][idx_to_select]

    x_tr_ds = np.vstack((x_tr_s, x_tr_b))
    y_tr_ds = np.vstack((y_tr_s, y_tr_b))

    assert x_tr_ds.shape[0] == nb_s * 2
    assert len(y_tr_ds) == nb_s * 2

    return x_tr_ds, y_tr_ds


#
# GRID SEARCH
#


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]
