"""
@author KillianRaude
@author JorisMonnet
@author ColinPelletier
"""
import numpy as np
import math
from helpers import batch_iter, build_poly


#
#   MANDATORY FUNCTIONS
#


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        w -= gamma * compute_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            w -= gamma * compute_gradient(y_batch, tx_batch, w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """ "Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    N, D = tx.shape
    w = np.linalg.solve(tx.T @ tx + lambda_ * 2 * N * np.eye(D), tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1})"""
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with regularization term λ∥w∥²)"""
    treshold = 1e-8
    lossList = []
    w = initial_w

    for _ in range(max_iters):
        lossList.append(compute_log_loss(y, tx, w, lambda_))
        w -= gamma * compute_gradient_sig(y, tx, w, lambda_)
        if len(lossList) > 1 and np.abs(lossList[-1] - lossList[-2]) < treshold:
            break  # converge criterion

    return w, lossList[-1]  # take only the last loss


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """

    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    te_indice = k_indices[k]
    x_tr = x[tr_indice]
    x_te = x[te_indice]
    y_tr = y[tr_indice]
    y_te = y[te_indice]

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w = ridge_regression(y_tr, tx_tr, lambda_)

    loss_tr = math.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = math.sqrt(2 * compute_mse(y_te, tx_te, w))
    return loss_tr, loss_te


#
#   HELPER FUNCTIONS
#


def compute_gradient(y, tx, w):
    return -tx.T.dot(y - tx @ w) / len(y)


def compute_mse(y, tx, w):
    return 0.5 * np.mean((y - tx @ w) ** 2)


def compute_log_loss(y, tx, w, lambda_):
    """Compute loss with log"""
    return (
        np.sum(np.log(1 + np.exp(tx @ w)))
        - (y.T @ (tx @ w))
        + lambda_ * (np.linalg.norm(tx) ** 2) / 2
    )


def compute_gradient_sig(y, tx, w, lambda_):
    """Gradient with sigmoid"""
    return tx.T @ (sigmoid(tx @ w) - y) + lambda_ * w


def sigmoid(t):
    """Sigmoid function"""
    return 1.0 / (1 + np.exp(-t))
