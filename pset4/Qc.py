import numpy as np
import random
from matplotlib import pyplot as plt
from scipy import spatial, stats

np.random.seed(2020)

def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def my_proposal(theta_cur):
    # Fixed Gaussian proposal.
    np.random.seed(2020)
    diag_terms = np.ones(len(theta_cur)) * 1e2
    Sigma_prop = np.diag(diag_terms)
    theta_prop = stats.multivariate_normal.rvs(theta_cur, Sigma_prop)
    cur_prop_pdf = stats.multivariate_normal.pdf(theta_cur, mean=theta_prop, cov=Sigma_prop)
    prop_cur_pdf = stats.multivariate_normal.pdf(theta_prop, mean=theta_cur, cov=Sigma_prop)
    return theta_prop, cur_prop_pdf, prop_cur_pdf

def  my_likelihood(y, theta):
    # Assumes additive Gaussian noise term with zero mean and standard deviation sigma.
    np.random.seed(2020)
    sigma = 1e0

    # Forward model computation.
    G = theta[0]**3 + theta[1]*np.exp(0.8)

    # Compute the likelihood PDF (using PDF of epsilon).
    likelihood = stats.norm.pdf(G-y, 0, sigma)
    return likelihood

def my_prior(theta):
    # Gaussian prior here.
    np.random.seed(2020)
    mu = [0, 0]
    Sigma = np.diag([1e0, 1e0])

    # Compute the prior PDF.
    prior = stats.multivariate_normal.pdf(theta, mu, Sigma)
    return prior




# Setting.
n_iters = 50000
burnin = 1
theta_true = [0.4, 1.3]
y = theta_true[0]**3 + theta_true[1]*np.exp(0.8) + 1e0*np.random.random()  # Our noisy data.
#theta_cur = mvnrnd([0 0], diag([1 1])); % Chain initial point.
theta_cur = [1, 1]  # Chain initial point.

theta_hist = np.zeros((n_iters, 2))
theta_hist[0, :] = theta_cur

for k in range(1, n_iters):
    # Generate a proposal point.
    [theta_prop, cur_prop_pdf, prop_cur_pdf] = my_proposal(theta_cur)
    # print(theta_prop, cur_prop_pdf, prop_cur_pdf, "****")
    theta_prop = [14.1541, 1.2033]
    cur_prop_pdf = 6.6988e-04
    prop_cur_pdf = 6.6988e-04
# Compute alpha
    alpha = min(1, (my_likelihood(y, theta_prop) * my_prior(theta_prop) * cur_prop_pdf) / ( my_likelihood(y, theta_cur) * my_prior(theta_cur) * prop_cur_pdf))
    print(alpha)
    break
    u = np.random.random()
    if (u < alpha):
        # Accept with probability alpha.
        theta_hist[k, :] = theta_prop
        theta_cur = theta_prop
    else:
        theta_hist[k, :] = theta_cur

print(theta_prop)

theta_plot = theta_hist[burnin:, :]


plt.plot(theta_plot[0:2000, 0], theta_plot[0:2000, 1], '*')
plt.xlabel('Theta1')
plt.ylabel('Theta2')
plt.show()
# print-depsc 'figures/scatter.eps'

plt.plot(theta_plot[0:10000, 0], '-')
plt.xlabel('Iter after burn-in')
plt.ylabel('Theta1')
plt.show()
# print -depsc 'figures/theta1.eps'


plt.plot(theta_plot[0:10000, 1], '-')
plt.xlabel('Iter after burn-in')
plt.ylabel('Theta2')
plt.show()
# print -depsc 'figures/theta2.eps'

ac1 = autocorr(theta_hist[:,1], 200)

plt.plot(ac1, '-', 'linewidth', 2)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
# print -depsc 'figures/ac1.eps'

ac2 = autocorr(theta_hist[:,2], 200)

plt.plot(ac2, '-', 'linewidth', 2)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
# print -depsc 'figures/ac2.eps'