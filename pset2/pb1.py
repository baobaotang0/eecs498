import numpy as np
from scipy import linalg
from matplotlib import pyplot


def Cholesky_factorization_solver(X, y):
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    R = np.linalg.cholesky(X.T @ X).T
    omega = linalg.solve_triangular(R.T, X.T @ y, lower=True)
    beta_hat = linalg.solve_triangular(R, omega, lower=False)
    return beta_hat


def SVD_solver(X, y):
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    omega = np.linalg.inv(np.diag(s)) @ (U.T @ y)
    beta_hat = Vh.T @ omega
    return beta_hat


def ridge_regression(X, y, l):
    y = np.array(y)
    y = y.reshape(len(y), 1)
    n = X.shape[1]
    X_hat = np.array(X, dtype='float64')
    for j in range(1, n):
        xj_avg = np.average(X[:, j])
        X_hat[:, j] -= xj_avg
    beta_1 = np.average(y)
    y_hat = np.vstack((y - beta_1, np.zeros((n, 1))))
    X_hat = np.vstack((X, np.eye(n) * np.sqrt(l)))
    beta_ridge = SVD_solver(X_hat, y_hat)
    return beta_ridge, beta_1


def get_X(z, deg):
    return np.array([[(z[i] ** j) for j in range(deg + 1)] for i in range(len(z))])


if __name__ == '__main__':
    z = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    y = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    x_show = np.linspace(-5, 5, 50)
    for deg in range(2, 11, 2):
        X = get_X(z, deg)
        beta_OLS = SVD_solver(X, y)
        pyplot.plot(x_show, np.poly1d(np.flip(beta_OLS.flatten()))(x_show),"--")
    pyplot.plot(z, y)
    pyplot.title("beta_OLS")
    pyplot.legend(["deg="+str(i) for i in range(2, 11, 2)])
    pyplot.show()

    for deg in range(2, 11, 2):
        X = get_X(z, deg)
        beta_ridge, beta_1 = ridge_regression(X, y, 0)
        pyplot.plot(x_show, np.poly1d(np.flip(beta_ridge.flatten()))(x_show) + beta_1, '--')
    pyplot.plot(z, y)
    pyplot.title("beta_ridge_regression(lambda=0)")
    pyplot.legend(["deg="+str(i) for i in range(2, 11, 2)])
    pyplot.show()

    for i in range(-1, 6):
        X = get_X(z, 10)
        beta_ridge, beta_1 = ridge_regression(X, y, 10**i)
        pyplot.plot(x_show, np.poly1d(np.flip(beta_ridge.flatten()))(x_show) + beta_1, '--')
    pyplot.plot(z, y)
    pyplot.title("beta_ridge_regression(deg=10)")
    deg = [0] + [10**i for i in range(-1, 6)]
    pyplot.legend(["lambda="+str(i) for i in deg])
    pyplot.show()

