# imports
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

#
#  import self-defined modules
# 
sys.path.append("../")
sys.path.append("../src/")

from src.helpers import *
from src.implementations import *
from src.cross_validation import *

#
# global variables
#
DATA_FOLDER = "./data/"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

RESULT_FOLDER = "./results/"
RESULT_FILE = "predictions.csv"

NAN_VALUE = -999.0
INTEGER_COLUMN = (
    22  # 24 in raw csv file, but 23 when id and prediction column are removed
)


#
# import dataset
#

# Load data
y_tr, x_tr, _ = load_csv_data(DATA_FOLDER + TRAIN_FILE, sub_sample=False)
y_te, x_te, ids_tests = load_csv_data(DATA_FOLDER + TEST_FILE, sub_sample=False)
print("x_tr shape : {}, y_tr shape : {}".format(x_tr.shape, y_tr.shape))
print("x_te shape : {}, y_te shape : {}".format(x_te.shape, y_te.shape))

# Define missing values as NAN
x_tr[x_tr == NAN_VALUE] = np.nan
x_te[x_te == NAN_VALUE] = np.nan

# Get columns names
col_names = []
with open(DATA_FOLDER + TRAIN_FILE) as dataset:
    col_names = dataset.readline().split(",")

#
# pre-processing
#

# before pre-processing
print("Before pre-processing:")
print("x_tr shape : {}".format(x_tr.shape))
print("x_te shape : {}".format(x_te.shape))
print("x_tr range :{} {}".format(np.nanmin(x_tr), np.nanmax(x_tr)))
print("x_te range :{} {}".format(np.nanmin(x_te), np.nanmax(x_te)))

# apply log transformation
cols_to_log_transform = ["DER_pt_h", "DER_pt_tot", "PRI_met", "PRI_met_sumet"]
cols_idx = [get_col_idx(col, col_names) for col in cols_to_log_transform]

x_tr, x_te = log_transform(x_tr, x_te, cols_idx)

# Remove columns with too much NAN
x_tr = remove_nan_columns(x_tr, 0.3)
x_te = remove_nan_columns(x_te, 0.3)

# Replace missing data by the mean
mean_x = np.nanvar(x_tr, axis=0)
x_tr = replace_nan_by_means(x_tr, mean_data=mean_x)
x_te = replace_nan_by_means(x_te, mean_data=mean_x)


assert x_tr[np.isnan(x_tr)].shape[0] == 0
assert x_te[np.isnan(x_te)].shape[0] == 0

# Standardize after replacing missing values
IDs_degrees = np.array([10, 13, 15])
x_tr = transform(x_tr, IDs_degrees)
x_te = transform(x_te, IDs_degrees)

# Polynomial feature expansion
xt_tr = build_poly(x_tr, 8)

# plot features after pre-processing
print("After pre-processing:")
print("x_tr shape : {}".format(x_tr.shape))
print("x_te shape : {}".format(x_te.shape))
print("x_tr range :{} {}".format(np.min(x_tr), np.max(x_tr)))
print("x_te range :{} {}".format(np.min(x_te), np.max(x_te)))

#
# model fitting
#
w, loss = ridge_regression(y_tr, xt_tr, lambda_=0.0005)
y_pred_tr = predict_reg(w, xt_tr)

print("Training accuracy : {}".format(accuracy(y_tr, y_pred_tr)))
print("Training f1       : {}".format(f1_score(y_tr, y_pred_tr)))

#
# predict test values and generate submissions
#

# Run on the test set
xt_te = build_poly(x_te, 8)
y_pred_te = predict_reg(w, xt_te)

y_test, input_test, ids_test = load_csv_data(DATA_FOLDER + TEST_FILE, False)

create_csv_submission(ids_test, y_pred_te, RESULT_FOLDER + RESULT_FILE)