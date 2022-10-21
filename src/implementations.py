"""
@author KillianRaude
@author JorisMonnet
@author ColinPelletier
"""
import numpy as np
from helpers import batch_iter


#
#   MANDATORY FUNCTIONS
#


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        w -= gamma * compute_gradient_ridge(y, tx, w)
    return w, compute_mse(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            w -= gamma * compute_gradient_ridge(y_batch, tx_batch, w)
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
    # init parameters
    threshold = 1e-8  # min difference improvement between two iterations
    losses = []
    w = initial_w

    # start the logistic regression
    for _ in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma, lambda_)

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]


#
# Cross validation
#


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, initial_w, max_iters, gamma):
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

    # w, loss = least_squares(y_tr, x_tr
    w, _ = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)

    # TODO update
    loss_tr = compute_log_loss(y_tr, x_tr, w, 0)
    loss_te = compute_log_loss(y_te, x_te, w, 0)

    return loss_tr, loss_te


def run_cross_validation(y, x, k_fold, initial_w, max_iters, gamma, seed=1):
    """cross validation over regularisation parameter lambda.

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """

    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []

    # run k predictions
    for k in range(k_fold):
        loss = cross_validation(
            y, x, build_k_indices(y, k_fold, seed), k, initial_w, max_iters, gamma
        )
        loss_tr.append(loss[0])
        loss_te.append(loss[1])

    mean_loss_tr = np.array(loss_tr).mean()
    mean_loss_te = np.array(loss_te).mean()

    return mean_loss_tr, mean_loss_te


#
#   REGRESSION
#


def compute_gradient_ridge(y, tx, w):
    return -tx.T.dot(y - tx @ w) / len(y)


def compute_mse(y, tx, w):
    return np.mean((y - tx @ w) ** 2)


#
#   LOGITSTIC REGRESSION
#


def compute_log_loss(y, tx, w, lambda_):
    """Compute loss with log"""
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    sigmoid_pred = sigmoid(tx @ w)
    sum_parts = y * np.log(sigmoid_pred) + (1 - y) * np.log(1 - sigmoid_pred)
    return -np.mean(sum_parts) + lambda_ * np.linalg.norm(w) ** 2


def compute_gradient_sig(y, tx, w, lambda_):
    """Gradient with sigmoid"""
    return 1 / tx.shape[0] * tx.T @ (sigmoid(tx @ w) - y) + lambda_ * w * 2


def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    loss = compute_log_loss(y, tx, w, lambda_)
    w = w - gamma * compute_gradient_sig(y, tx, w, lambda_)

    return loss, w


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))
