import numpy as np
from matplotlib import pyplot as plt


def G(theta, d):
    return (theta ** 3) * (d ** 2) + theta * np.exp(-np.abs(0.2 - d))


def norm_logpdf(x, loc=0.0, scale=1.0):
    return -np.log(np.sqrt(2 * np.pi) * scale) - (x - loc) ** 2 / 2 / scale ** 2


def norm_pdf(x, loc=0.0, scale=1.0):
    return np.exp(-(x-loc)**2/2/scale**2)/scale/np.sqrt(2*np.pi)


def get_one_utility(d, theta_i, theta_ij, noises, sigma, model_fun):
    N1, N2 = len(theta_i), len(theta_ij)

    return

def get_utility(num_d, N1, N2, model_fun=G, sigma=0.01):
    d_set = np.linspace(0, 1, num_d)
    U = np.zeros((num_d,))

    theta_i = np.random.uniform(size=(N1,))
    theta_ij = np.random.uniform(size=(N2,))
    noise = np.random.normal(size=(N1,))

    for k, d in enumerate(d_set):
        G_i, G_ij = model_fun(theta_i.reshape((N1, 1)), d), model_fun(theta_ij.reshape((N2, 1)), d)
        ys = G_i + sigma * noise.reshape((N1, 1))
        log_likelihood = norm_logpdf(ys, G_i, sigma)
        evidence = np.array([np.mean(norm_pdf(ys[i:i + 1], G_ij, sigma)) for i in range(N1)])
        # teX = np.concatenate((theta_i, np.ones((N1, 1))*d), axis=1)
        # y[:, k] = model.predict(teX).flatten() + noise[:, k]
        U[k] = np.mean(log_likelihood - np.log(evidence))
    return d_set, U

if __name__ == '__main__':
    d_set, U = get_utility(30, 200, 200)

    plt.plot(d_set, U)
    plt.show()

