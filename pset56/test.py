import numpy as np
import matplotlib.pyplot as plt


def norm_logpdf(x, loc=0.0, scale=1.0):
    return -np.log(np.sqrt(2 * np.pi) * scale) - (x - loc) ** 2 / 2 / scale ** 2


def norm_pdf(x, loc=0.0, scale=1.0):
    return np.exp(-(x - loc) ** 2 / 2 / scale ** 2) / scale / np.sqrt(2 * np.pi)


def G(theta, d):
    return theta ** 3 * d ** 2 + theta * np.exp(-np.abs(0.2 - d))


def utility(d, theta_i, theta_ij, noises, sigma):
    n_sample = len(theta_i)
    G_i = G(theta_i, d)
    G_ij = G(theta_ij, d)
    ys = G_i + sigma * noises
    loglikelis = norm_logpdf(ys, G_i, sigma)
    # plt.plot(loglikelis)
    evids = np.array([np.mean(norm_pdf(ys[i:i + 1], G_ij, sigma)) for i in range(n_sample)])
    return np.mean(loglikelis - np.log(evids))

if __name__ == '__main__':
    num_d = 30
    sigma = 0.01
    N1, N2 = 200, 200

    d_set = np.linspace(0, 1, num_d)

    Us = []
    theta_i = np.random.uniform(size=(N1, 1))
    theta_ij = np.random.uniform(size=(N2, 1))
    noises = np.random.normal(size=(N1, 1))
    for d in d_set:
        Us.append(utility(d, theta_i, theta_ij, noises, sigma))

    y = np.zeros((N1, num_d))
    U = np.zeros((num_d,))
    noise = np.random.normal(size=(N1, 1))*sigma

    for k, d in enumerate(d_set):
        # teX = np.concatenate((theta_i, np.ones((N1, 1))*d), axis=1)
        # y[:, k] = model.predict(teX).flatten() + noise[:, k]
        G_i = G(theta_i, d)
        G_ij = G(theta_ij, d)
        y[:, k] = G_i + noise
        log_likelihood = norm_logpdf(y[:, k].reshape((N1,1)), G_i, sigma)
        evidence = np.array([np.mean(norm_pdf(y[i, k], G_ij, sigma)) for i in range(N1)])
        U[k] = np.mean(log_likelihood - np.log(evidence))

    # plt.plot(log_likelihood)
    # plt.show()

    plt.plot(d_set, U)
    plt.show()
    #
    # plt.figure(figsize=(6, 4))
    # plt.plot(d_set, Us)
    # plt.xlabel('d', fontsize=20)
    # plt.ylabel('U(d)', fontsize=20)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.grid(ls='--')
    # plt.show()


