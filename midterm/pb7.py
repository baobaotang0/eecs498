import pandas as pd
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np

boston = load_boston()
X = boston.data
y = boston.target

# Scale datasets
X = preprocessing.scale(X)

# The number of features
d = X.shape[1]

# Split the data into train (including validation data) and test
n_train_all = 400
n_val = 100

X_train_all = X[0:n_train_all]
y_train_all = y[0:n_train_all]
X_val = X[0:n_val]
y_val = y[0:n_val]
X_train = X[n_val:n_train_all]
y_train = y[n_val:n_train_all]
X_test = X[n_train_all:]
y_test = y[n_train_all:]
# Set up the available parameter grids, you should not change this
alpha_grid = np.linspace(0.9, 1, 100)
lambda_grid = np.linspace(0.01, 20, 100)

# Create an all-zero matrix to store the validation error
val_err = np.zeros((len(alpha_grid), len(lambda_grid)))

# The runtime of this nested for loop should be less than 20 seconds if everything is correct
for i in range(len(alpha_grid)):
    for j in range(len(lambda_grid)):
        alp = alpha_grid[i]
        lam = lambda_grid[j]
        ##################################################
        # TODO: Express beta in terms of alpha and lambda#

        beta = lam * (1 - alp)

        ##################################################

        reg = linear_model.Lasso(alpha=beta)

        ##################################################
        # TODO: Construct the \tilde{X} and \tilde{y}    #        print(X_train.shape)

        X_train_tilde = np.concatenate((X_train, np.sqrt(lam*alp)*np.eye(X_train.shape[1])), axis=0)
        y_train_tilde = np.concatenate((y_train, np.zeros(X_train.shape[1])), axis=0)
        X_val_tilde = np.concatenate((X_val, np.sqrt(lam*alp)*np.eye(X_val.shape[1])), axis=0)
        y_val_tilde = np.concatenate((y_val, np.zeros(X_train.shape[1])), axis=0)

        ##################################################

        reg.fit(X_train_tilde, y_train_tilde)

        ##################################################
        # TODO: Calculate the validation errors with the #
        # current parameters and store into val_err.     #

        val_err_ij = (np.linalg.norm(reg.predict(X_val_tilde) - y_val_tilde)**2)/len(y_val_tilde)

        ##################################################

        val_err[i, j] = val_err_ij
# TODO: Find the indices of best parameter pairs, which gives the lowest val error
opt = np.argmin(val_err)
ind = [opt // len(alpha_grid), opt % len(lambda_grid)]  # Try np.unravel_index() or just use np.argmin() shown in HW3 Pr2 (b)

# TODO: Pick the best parameters in the given range
alpha_best = alpha_grid[ind[0]]
lambda_best = lambda_grid[ind[1]]
beta_best = lambda_best * (1 - alpha_best)

# TODO: Retrain the LASSO model with best parameters and all training data
reg = linear_model.Lasso(alpha=beta_best)
X_train_all_tilde = np.concatenate((X_train_all, np.sqrt(lambda_best*alpha_best)*np.eye(X_train_all.shape[1])), axis=0)
y_train_all_tilde = np.concatenate((y_train_all, np.zeros(X_train_all.shape[1])), axis=0)
reg.fit(X_train_all_tilde, y_train_all_tilde)

# TODO: Calculate the MSE on test datasets with the best LASSO model
y_test_pred = reg.predict(X_test)
mse_test = (np.linalg.norm(y_test_pred - y_test)**2)/len(y_test)

print(alpha_best, lambda_best)
print("The optimal theta: ", reg.coef_)
print("The MSE on test datasets: ", mse_test)
