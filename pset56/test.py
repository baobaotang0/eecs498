import numpy as np
import matplotlib.pyplot as plt


def norm_logpdf(x, loc=0.0, scale=1.0):
    return -np.log(np.sqrt(2 * np.pi) * scale) - (x - loc) ** 2 / 2 / scale ** 2


def norm_pdf(x, loc=0.0, scale=1.0):
    return np.exp(-(x - loc) ** 2 / 2 / scale ** 2) / scale / np.sqrt(2 * np.pi)


def G(theta, d):
    return theta ** 3 * d ** 2 + theta * np.exp(-np.abs(0.2 - d))


def utility(d, thetas, noises, sigma):
    n_sample = len(thetas)
    Gs = G(thetas, d)
    # print("Gs.shape",Gs.shape)
    ys = Gs + sigma * noises
    # print("ys", ys.shape, self.noise_loc, sigma , self.noise_r_s)
    loglikelis = norm_logpdf(ys, Gs, sigma)
    evids = np.array([np.mean(norm_pdf(ys[i:i + 1], Gs, sigma)) for i in range(n_sample)])
    return np.mean(loglikelis - np.log(evids))


noise_base_scale = 0.01

ds = np.linspace(0, 1, 21)
Us = []
thetas = np.random.uniform(size=(1000, 1))
noises = np.random.normal(size=(1000, 1))
for d in ds:
    Us.append(utility(d, thetas, noises, noise_base_scale))

plt.figure(figsize=(6, 4))
plt.plot(ds, Us)
plt.xlabel('d', fontsize=20)
plt.ylabel('U(d)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(ls='--')
plt.show()
