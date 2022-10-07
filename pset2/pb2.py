import numpy as np
from matplotlib import pyplot as plt

def read_txt(address):
    with open(address,"r") as f:
        text = f.readlines()
        z, y= np.zeros((len(text))), np.zeros((len(text)))
        for i, line in enumerate(text):
            z[i], y[i] = line.split()
        return z, y

def get_X(z):
    return np.array([[(z[i] ** j) for j in range(6)] +
                     [np.sin(2*np.pi*0.5*f*z[i]) for f in range(1, 9)] +
                     [np.cos(2*np.pi*0.5*f*z[i]) for f in range(1, 9)] +
                     [np.exp(z[i])] +
                     [np.log(z[i])]
                     for i in range(len(z))])

def SVD_solver(X, y):
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(len(y), 1)
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    omega = np.linalg.inv(np.diag(s)) @ (U.T @ y)
    beta_hat = Vh.T @ omega
    return beta_hat


if __name__ == '__main__':
    name = ["z^" + str(j) for j in range(6)] + \
           ["cos(" + str(f)+"πz)" for f in range(1, 9)] + \
           ["sin(" + str(f) + "πz)" for f in range(1, 9)] + \
           ["exp(z)", "In(z)"]

    z_a, y_a = read_txt("pset2_files/data1.txt")
    X_a = get_X(z_a)
    beta_a = SVD_solver(X_a, y_a).flatten()
    plt.bar(name,height=beta_a)
    plt.grid("on")
    plt.show()


