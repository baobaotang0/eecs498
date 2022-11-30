"""
Script for running GMM soft clustering
"""

import matplotlib.pyplot as plt
import numpy as np
import string as s
from scipy.stats import multivariate_normal

from sklearn.datasets import fetch_openml

from gmm import gmm


def get_data():
    """Load penguins data from Github."""
    penguins = fetch_openml("penguins", as_frame=False)
    # get 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm',
    # 'body_mass_g' features
    X = penguins["data"][:, 1:5]
    # drop NA values
    X = X[~np.isnan(X).any(axis=1)]
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    print(f"Shape of the input data: {X.shape[0]} by {X.shape[1]}")
    return X


def main():
    """Call GMM with different numbers of clusters.

    - num_K is an array containing the tested cluster sizes
    - cluster_proportions maps each cluster size to a size by 1 vector
      containing the mixture proportions
    - means is a dictionary mapping the cluster size to matrix of means
    - z_K maps each cluster size into a num_points by k matrix of pointwise
      cluster membership probabilities
    - sigma2 maps each cluster size to the corresponding sigma^2 value learnt
    - BIC_K contains the best BIC values for each of the cluster sizes
    """
    print(
            "We'll try different numbers of clusters with GMM, using multiple runs"
            " for each to identify the 'best' results"
    )
    np.random.seed(445)
    trainX = get_data()
    num_K = range(2, 9)  # List of cluster sizes
    BIC_K = np.zeros(len(num_K))

    xVals = trainX[:, 0]
    yVals = trainX[:, 1]
    x = np.linspace(np.min(xVals), np.max(xVals), 500)
    y = np.linspace(np.min(yVals), np.max(yVals), 500)
    X, Y = np.meshgrid(x, y)
    pos = np.array([X.flatten(), Y.flatten()]).T

    for idx in range(len(num_K)):
        # Running
        k = num_K[idx]
        print("%d clusters..." % k)
        # TODO: Run gmm function 10 times and get the best set of parameters
        # for this particular value of k. Use the default num_iter=10 in calling gmm()
        for i in range(10):
            mu, pk, zk, si2, BIC = gmm(trainX[:, :2], k, num_iter=30, plot=False)
            if BIC_K[idx] == 0 or BIC < BIC_K[idx]:
                BIC_K[idx] = BIC
                best_mu, best_si2 = mu, si2

    # TODO: Part d: Make a plot to show BIC as function of clusters K

        plt.clf()
        plt.scatter(xVals, yVals, color="black")
        pdfs = []
        for i in range(k):
            rv = multivariate_normal(best_mu[i], best_si2)
            plt.plot(best_mu[i][0], best_mu[i][1], "*r")
            pdfs.append(rv.pdf(pos).reshape(500, 500))
        pdfs = np.array(pdfs)
        plt.contourf(X, Y, np.max(pdfs, axis=0), alpha=0.8)

        plt.savefig(f"bic_plot_{k}.png")


if __name__ == "__main__":
    main()
