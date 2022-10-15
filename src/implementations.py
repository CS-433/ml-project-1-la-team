"""
@author KillianRaude
@author JorisMonnet
@author ColinPelletier
"""
import numpy as np

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
    #return w, loss

#
#   HELPER FUNCTIONS
# 

def compute_gradient(y,tx,w):
    return - tx.T.dot(y - tx @ w) / len(y)

def compute_mse(y,tx,w):
    return 0.5*np.mean((y - tx @ w)**2)

def replace_nan_by_means(dataset):
    """

        Test :  arr_test = np.array([[1, 2, 3, 4], [10, np.nan, 11, 12], [np.nan, 13, 14, np.nan], [np.nan, 15, 16, 17]])
                arr_test_theoric = replace_nan_by_means(arr_test)
                assert(np.allclose(arr_test_theoric[1, 1], np.nanmean(arr_test[:, 1]))) #, "mean not computed correctly"
    """
    def replace_nan_by_feature_mean(feature):
        """
            input : a columns of the dataset
            
        """
        feature[np.isnan(feature)] = np.nanmean(feature)
        return feature

    return np.apply_along_axis(replace_nan_by_feature_mean, 0, dataset)