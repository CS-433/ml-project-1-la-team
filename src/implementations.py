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
    treshold = 1e-8
    lossList = []
    w = initial_w

    for _ in range(max_iters):
        lossList.append(compute_log_loss(y, tx, w, lambda_))
        w -= gamma * compute_gradient_sig(y,tx,w,lambda_)
        if len(lossList) > 1 and np.abs(lossList[-1] - lossList[-2]) < treshold:
            break #converge criterion

    return w, lossList[-1]

#
#   HELPER FUNCTIONS
# 

def compute_gradient(y,tx,w):
    return - tx.T.dot(y - tx @ w) / len(y)

def compute_mse(y,tx,w):
    return 0.5*np.mean((y - tx @ w)**2)

def compute_log_loss(y,tx,w,lambda_):
    return np.sum(np.log(1 + np.exp(tx @ w))) - (y.T @ (tx @ w)) + lambda_ * (np.linalg.norm(tx) ** 2) / 2

def compute_gradient_sig(y,tx,w,lambda_):
    """Gradient with sigmoid"""
    return tx.T @ (sigmoid(tx @ w) - y) + lambda_ * w

def sigmoid(t):
    """Sigmoid function"""
    return 1.0 / (1 + np.exp(-t))