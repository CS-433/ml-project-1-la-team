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


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        w -= gamma * compute_gradient_ridge(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
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

    # TODO remove it
    best_w = w
    best_loss = 1e6

    # start the logistic regression
    for _ in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma, lambda_)

        # select the max w
        if loss < best_loss and loss > 0:
            best_w = w
            best_loss = loss

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("===== converges ! =========")
            break

    return w, compute_log_loss(y, tx, w), best_w, best_loss


# TODO select only one
def reg_logistic_regression_sgd(y, tx, lambda_, initial_w, max_iters, gamma):
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
            print("===== converges ! =========")
            break

    return w, compute_log_loss(y, tx, w)


#
#   REGRESSION
#


def compute_gradient_ridge(y, tx, w):
    """Compute Gradient Ridge"""
    return -tx.T.dot(y - tx @ w) / len(y)


def compute_mse(y, tx, w):
    """Compute MSE with 1/2 factor"""
    return 0.5 * np.mean((y - tx @ w) ** 2)


#
#   LOGITSTIC REGRESSION
#


def compute_log_loss(y, tx, w):
    """Compute loss with log"""
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    sigmoid_pred = sigmoid(tx @ w)
    sum_parts = y * np.log(sigmoid_pred) + (1 - y) * np.log(1 - sigmoid_pred)
    return -np.mean(sum_parts)


def compute_gradient_sig(y, tx, w, lambda_):
    """Compute gradient with sigmoid"""
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
    loss = compute_log_loss(y, tx, w)
    w -= gamma * compute_gradient_sig(y, tx, w, lambda_)

    return loss, w


def sigmoid(t):
    """Compute sigmoid"""
    return 1.0 / (1.0 + np.exp(-t))