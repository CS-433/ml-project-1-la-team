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
    for n_iter in range(max_iters):
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

        # TODO remove it
        print("SGD iter. {bi}/{ti}: loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

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


def cross_validation(y, x, k_indices, k, lambda_, is_regression, initial_w=None, degree=1, max_iters=0, gamma=0, threshold=0):
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

    w = 0
    loss_tr = {}
    loss_te = {}

    # add polynomial degree
    xt_tr = build_poly(x_tr, degree) # if degree=1, add the offset
    xt_te = build_poly(x_te, degree)

    if is_regression:
        # w, loss_train = ridge_regression(y_tr, xt_tr, lambda_)
        w, loss_train = ridge_regression(y_tr, xt_tr, lambda_)

        y_pred_tr = predict_reg(w, xt_tr, threshold=threshold)
        y_pred_te = predict_reg(w, xt_te, threshold=threshold)
        loss_test = compute_mse(y_te, xt_te, w)
    else:
        w, loss_train = reg_logistic_regression(y_tr, xt_tr, lambda_, initial_w, max_iters, gamma)
        print("fold {} loss_tr {}".format(k, loss_train))

        y_pred_tr = predict_log(w, xt_tr)
        y_pred_te = predict_log(w, xt_te)
        loss_test = compute_log_loss(y_te, xt_te, w)
        print("fold {} loss_tr {}".format(k, loss_train))

    # compute scores
    loss_tr['acc'] = accuracy(y_tr, y_pred_tr)
    loss_tr['f1'] = f1_score(y_tr, y_pred_tr)
    loss_tr['loss'] = loss_train
    
    loss_te['acc'] = accuracy(y_te, y_pred_te)
    loss_te['f1'] = f1_score(y_te, y_pred_te)
    loss_te['loss'] = loss_test

    return loss_tr, loss_te

def run_cross_validation(y, x, k_fold, is_regression, lambdas=[0.0], gammas=[0.0], initial_w=None, degrees=[1], max_iters=0, threshold=0, seed=2):
    """cross validation over regularisation parameter lambda.

    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        is_regression: boolean
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    res = []
    best_res = {
                    'lambda': lambdas[0],
                    'gamma': gammas[0],
                    'degree': degrees[0],
                    'acc': -1,
                    'f1': -1
                }

    for gamma in gammas:
        for lambda_ in lambdas:
            for degree in degrees:
                print("lambda_={}, degree={}, gamma={}".format(lambda_, degree, gamma)) # TODO remove
                # define lists to store the loss of training data and test data
                k_fold_res = {
                    'lambda': lambda_,
                    'gamma': gamma,
                    'degree': degree
                }

                k_fold_res_tr_acc = []
                k_fold_res_te_acc = []

                k_fold_res_tr_f1 = []
                k_fold_res_te_f1 = []

                k_fold_res_tr_loss = []
                k_fold_res_te_loss = []

                # run k predictions
                for k in range(k_fold):
                    loss_tr, loss_te = cross_validation(y, x, build_k_indices(y, k_fold, seed), k, lambda_, is_regression, initial_w, degree, max_iters, gamma)

                    k_fold_res_tr_acc.append(loss_tr['acc'])
                    k_fold_res_te_acc.append(loss_te['acc'])

                    k_fold_res_tr_f1.append(loss_tr['f1'])
                    k_fold_res_te_f1.append(loss_te['f1'])
                    
                    k_fold_res_tr_loss.append(loss_tr['loss'])
                    k_fold_res_te_loss.append(loss_te['loss'])

                # add results
                k_fold_res['tr'] = {
                        'acc': np.array(k_fold_res_tr_acc).mean(),
                        'f1': np.array(k_fold_res_tr_f1).mean(),
                        'loss': np.array(k_fold_res_tr_loss).mean()
                    }

                k_fold_res['te'] = {
                        'acc': np.array(k_fold_res_te_acc).mean(),
                        'f1': np.array(k_fold_res_te_f1).mean(),
                        'loss': np.array(k_fold_res_te_loss).mean()
                    }

                # set the best result
                if k_fold_res['te']['acc'] > best_res['acc']:
                    best_res['lambda'] = lambda_
                    best_res['gamma'] = gamma
                    best_res['degree'] = degree
                    best_res['acc'] = k_fold_res['te']['acc']
                    best_res['f1'] = k_fold_res['te']['f1']

                res.append(k_fold_res)

    return res, best_res


#
#   REGRESSION
#


def compute_gradient_ridge(y, tx, w):
    return -tx.T.dot(y - tx @ w) / len(y)


def compute_mse(y, tx, w):
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
    loss = compute_log_loss(y, tx, w)
    w -= gamma * compute_gradient_sig(y, tx, w, lambda_)

    return loss, w


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


#
# GRID SEARCH
#


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def balance_dataset(x_tr, y_tr):
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
    
    assert(x_tr_ds.shape[0] == nb_s*2)
    assert(len(y_tr_ds) == nb_s*2)

    return x_tr_ds, y_tr_ds

def predict_log(w,x):
    """"
    
    """
    assert(w.shape[0] == x.shape[1])
    y_pred = sigmoid(x @ w)

    y_pred[y_pred <= 0.5] = -1
    y_pred[y_pred > 0.5] = 1

    return y_pred

def predict_reg(w, x, threshold=0.0):
    """"
        Add lambda_ to the prediction
    """
    assert(w.shape[0] == x.shape[1])
    y_pred = x @ w

    y_pred[y_pred <= threshold] = -1
    y_pred[y_pred > threshold] = 1

    return y_pred

def remove_nan_columns(x, max_nan_ratio=0.5):
    """
    
    """
    nb_nan = np.count_nonzero(np.isnan(x), axis=0)
    nan_ratio = nb_nan / x.shape[1]
    
    x = x[:, nan_ratio <= max_nan_ratio]
    return x

#
# Classification metrics
#
def accuracy(y_true, y_pred):
    """Compute accuracy"""
    return np.sum(y_true == y_pred) / len(y_true)

def f1_score(y_true, y_pred):
    """Compute f1 score"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + 0.5 * (fp + fn))

def add_offset(x):
    """ """
    return np.hstack((np.ones(x.shape[0])[:, np.newaxis], x))


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly
    