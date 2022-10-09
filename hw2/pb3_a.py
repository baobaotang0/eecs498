from sklearn.datasets import load_boston
from sklearn import preprocessing
import numpy as np

# load the boston dataset as a bunch (dictionary-like
# container object used by sklearn)
boston = load_boston()
# access the data and targets
X = boston.data
y = boston.target
X_train = X[0:400]
y_train = y[0:400]
X_test = X[400:]
y_test = y[400:]


def SVD_solver(X, y):
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    omega = np.linalg.inv(np.diag(s)) @ (U.T @ y)
    beta_hat = Vh.T @ omega
    return beta_hat


def get_X(X):
    X = preprocessing.scale(X)
    return np.hstack((np.ones((X.shape[0], 1)), X))


if __name__ == '__main__':
    theta_hat = SVD_solver(X=get_X(X_train), y=y_train)
    MSE = np.linalg.norm(y_test - get_X(X_test) @ theta_hat) ** 2 / len(y_test)
    print(MSE)
    from matplotlib import pyplot as plt
    plt.plot(y_test)
    plt.plot(get_X(X_test) @ theta_hat)
    plt.show()