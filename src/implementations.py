"""
@author KillianRaude
@author JorisMonnet
@author ColinPelletier
"""
import numpy as np
from helpers import batch_iter

def least_squares_GD(y, tx, initial_w,max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        w -= gamma * compute_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)

def least_squares_SGD(y, tx, initial_w,max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            w -= gamma * compute_gradient(y_batch, tx_batch, w)
    return w, compute_mse(y, tx, w)

def least_squares(y, tx) :
    """"Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    return w, compute_mse(y,tx,w)

def ridge_regression(y, tx, lambda_) :
    """Ridge regression using normal equations"""
    N, D = tx.shape
    w = np.linalg.solve(tx.T @ tx + lambda_*2*N*np.eye(D), tx.T@y)
    return w, compute_mse(y, tx, w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1})"""
    return reg_logistic_regression(y,tx,0,initial_w,max_iters,gamma)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with regularization term λ∥w∥²)"""
    #return w, loss


def compute_gradient(y,tx,w):
    - tx.T.dot(y - tx @ w) / len(y)

def compute_mse(y,tx,w):
    0.5*np.mean((y - tx @ w)**2)