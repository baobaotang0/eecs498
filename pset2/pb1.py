import numpy as np
from scipy import linalg


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
    omega = np.linalg.inv(np.diag(s))@(U.T @ y)
    beta_hat = Vh.T@omega
    return beta_hat



if __name__ == '__main__':
    X = [[1, 2, 3], [1, 5, 0], [5, 8, 9], [10, 0, 1]]
    y = [1, 2, 3, 4]
    print(Cholesky_factorization_solver(X, y))
    print(SVD_solver(X, y))
