import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial, stats

A = [[1, 2], [3, 4]]
b = [5, 6]
y = [4.786737690957183, 6.631541228821935]

surface_resolution = 200  # Resolution of the surface to plot
# lets create a grid of our two parameters
x1s = np.linspace(-3, 3, num=surface_resolution)
x2s = np.linspace(-3, 3, num=surface_resolution)
x1, x2 = np.meshgrid(x1s, x2s)

'''Q(a)'''
mu_a = [0.98813, -0.58643]
sigma_a = [[2001/43001,	-200/6143], [-200/6143,	 143/6143]]
post_a = np.zeros((surface_resolution, surface_resolution))
for i in range(surface_resolution):
    for j in range(surface_resolution):
        post_a[i, j] = stats.multivariate_normal.pdf(
            np.array([x1[i, j], x2[i, j]]),
            mean=mu_a, cov=sigma_a)

con1 = plt.contourf(x1s, x2s, post_a)
plt.xlabel("theta1")
plt.xlabel("theta2")
plt.title("the posterior PDF using  analytical solution.")
plt.axis('equal')
plt.show()

'''Q(b)'''
likelihood = np.zeros((surface_resolution, surface_resolution))
for i in range(surface_resolution):
    for j in range(surface_resolution):
        likelihood[i, j] = stats.multivariate_normal.pdf(
            np.array([y - A@np.array([x1[i, j], x2[i, j]]) - b]),
            mean=[0.0, 0.0], cov=0.01*np.eye(2))

prior = np.zeros((surface_resolution, surface_resolution))
for i in range(surface_resolution):
    for j in range(surface_resolution):
        prior[i, j] = stats.multivariate_normal.pdf(
            np.array([x1[i, j], x2[i, j]]),
            mean=[0.0, 0.0], cov=np.eye(2))

unnormalized_posterior = prior * likelihood
posterior = unnormalized_posterior / np.nan_to_num(unnormalized_posterior).sum()

con1 = plt.contourf(x1s, x2s, posterior)
plt.xlabel("theta1")
plt.xlabel("theta2")
plt.title("the posterior PDF using the “brute force” gridding method.")
plt.axis('equal')
plt.show()