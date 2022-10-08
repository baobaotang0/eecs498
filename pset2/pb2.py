import numpy as np
from matplotlib import pyplot as plt


def read_txt(address):
    with open(address, "r") as f:
        text = f.readlines()
        z, y = np.zeros((len(text))), np.zeros((len(text)))
        for i, line in enumerate(text):
            res = line.split()
            if res != []:
                z[i], y[i] = line.split()
            else:
                z, y = z[:i], y[:i]
                break
        return z, y


def get_X(z, p, f):
    return np.array([[(z[i] ** j) for j in range(p + 1)] +
                     [np.sin(2 * np.pi * 0.5 * f * z[i]) for f in range(1, f * 2 + 1)] +
                     [np.cos(2 * np.pi * 0.5 * f * z[i]) for f in range(1, f * 2 + 1)] +
                     [np.exp(z[i])] +
                     [np.log(z[i])]
                     for i in range(len(z))])


def get_param_name(p, f):
    return ["z^" + str(j) for j in range(p + 1)] + \
           ["cos(" + str(f) + "πz)" for f in range(1, f * 2 + 1)] + \
           ["sin(" + str(f) + "πz)" for f in range(1, f * 2 + 1)] + \
           ["exp(z)", "In(z)"]


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

def cross_validation(z, y, K, p, f, l):
    n = len(z)
    step = int(n / K)
    error = 0
    for i in range(K):
        z_hat = np.hstack((z[:i * step], z[(i + 1) * step:]))
        y_hat = np.hstack((y[:i * step], y[(i + 1) * step:]))
        z_valid = z[i * step: (i + 1) * step]
        y_valid = y[i * step: (i + 1) * step]
        X = get_X(z=z_hat, p=p, f=f)
        beta_ridge, beta_1 = ridge_regression(X=X, y=y_hat, l=l)
        y_test = get_X(z=z_valid, p=p, f=f) @ beta_ridge + beta_1
        error += np.linalg.norm(y_valid-y_test)/step
    return error



if __name__ == '__main__':
    p, f = 5, 4
    name_a = get_param_name(p, f)
    z_a, y_a = read_txt("pset2_files/data1.txt")
    X_a = get_X(z_a, p, f)
    beta_a = SVD_solver(X_a, y_a).flatten()
    plt.bar(name_a, height=beta_a)
    plt.grid("on")
    plt.show()
    plt.plot(z_a, y_a,"*-")
    z_show = np.linspace(min(z_a),max(z_a), 100)
    plt.plot(z_show, get_X(z=z_show, p=p, f=f) @ beta_a, "--")
    plt.legend(["data","fitting result"])
    plt.show()


    p, f = 6, 5
    name_b = get_param_name(p, f)
    z_b, y_b = read_txt("pset2_files/data2.txt")
    l = np.linspace(0.01, 0.1, 91)#[10**i for i in range(-10, 20)]
    error = [cross_validation(z_b, y_b, K=6, p=6, f=5, l=i) for i in l]
    l_final = l[error.index(min(error))]
    print(l_final)
    X_b = get_X(z=z_b, p=p, f=f)
    beta_ridge, beta_1 = ridge_regression(X=X_b, y=y_b, l=l_final)
    plt.bar(name_b, height=beta_ridge.flatten())
    plt.grid("on")
    plt.show()
    plt.plot(z_b, y_b, "*-")
    z_show = np.linspace(min(z_b),max(z_b), 500)
    plt.plot(z_show, get_X(z=z_show, p=p, f=f) @ beta_ridge + beta_1, "--")
    plt.legend(["data", "fitting result"])
    plt.show()

    p, f = 6, 5
    sigma = 5
    name_c = get_param_name(p, f)
    z_c, y_c = read_txt("pset2_files/data2.txt")
    br1 = np.arange(len(name_c))
    barwidth, line_num = 0.05, 2
    sigma_set = [0.2*i for i in range(8)]
    for i, sigma in enumerate(sigma_set):
        ax = plt.subplot(int(np.ceil(len(sigma_set)/line_num)), line_num, i+1)
        y_noisy = y_c + sigma * np.random.randn(len(y_c))
        l = [10**i for i in range(-10, 20)]
        error = [cross_validation(z_c, y_noisy, K=6, p=6, f=5, l=i) for i in l]
        l_final = l[error.index(min(error))]
        print(l_final)
        X_b = get_X(z=z_c, p=p, f=f)
        beta_ridge, beta_1 = ridge_regression(X=X_b, y=y_noisy, l=l_final)
    #     br1 = [x + barwidth for x in br1]
    #     plt.bar(br1, beta_ridge.flatten(), width=barwidth)
    # plt.xticks([r + barwidth for r in range(len(name_c))], name_c)
    # plt.legend([f'σ={sigma:.2}' for sigma in sigma_set], loc="lower right")
    # plt.grid("on")
    # plt.show()
        ax.plot(z_c, y_noisy,"*-")
        z_show = np.linspace(min(z_c),max(z_c), 200)
        ax.plot(z_show, get_X(z=z_show, p=p, f=f) @ beta_ridge +beta_1, "--")
        ax.legend(["data", "fitting result"])
        ax.set_title(f'σ={sigma:.2}', fontweight="bold", fontsize=10)
    plt.show()