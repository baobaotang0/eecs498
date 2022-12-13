from time import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import manifold, metrics
from sklearn.decomposition import PCA

FASHION_CLASS = ['0: T-shirt/top', '1: Trouser', '2: Pullover', '3: Dress', '4: Coat', '5: Sandal',
                 '6: Shirt', '7: Sneaker', '8: Bag', '9: Ankle boot']


def plot_embedding(X, y, t, plot=True):
    plt.clf()
    fig = plt.figure()
    if X.shape[1] == 3:
        dim = 3
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=0.5)
    elif X.shape[1] == 2:
        dim = 2
        ax = fig.add_subplot(1, 1, 1)
        p = ax.scatter(X[:, 0], X[:, 1], c=y, s=0.1)
    else:
        return
    ax.set_title(t)
    fig.colorbar(p).ax.set_yticklabels(FASHION_CLASS)
    plt.savefig(f"{t}_{dim}D.png")
    if plot:
        plt.show()


def plot_fashion_MINIST():
    fig = plt.figure()
    j = 0
    i = 0
    while i < 10:
        if y_train[j] == i:
            ax = fig.add_subplot(4, 3, 1 + i)
            ax.imshow(X_train[j])
            ax.set_title(FASHION_CLASS[y_train[j]])
            ax.get_xaxis().set_visible(False)
            i += 1
        j += 1
    plt.show()


if __name__ == '__main__':

    fashion_data = keras.datasets.fashion_mnist.load_data()
    (X_train, y_train), (X_test, y_test) = fashion_data
    X = X_test.reshape((10000, 28 * 28))
    y = y_test

    plt.set_cmap('gist_rainbow')
    rd_time = []
    plot_flag = False
    for line, dim in [0, 2], [1, 3]:
        rd_time.append([])
        for model in [PCA(n_components=dim), ]
            t_start = time()
            my_pca = PCA(n_components=dim)
            X_pca = my_pca.fit_transform(X)
            # plot_embedding(X_pca[:, 0:dim], y, 'PCA')
            rd_time[line].append(time() - t_start)

        t_start = time()
        my_MDS = manifold.MDS(n_components=dim, n_init=1, max_iter=100)
        X_mds = my_MDS.fit_transform(X)
        rd_time[line].append(time() - t_start)
        # plot_embedding(X_mds, y, 'MDS')

        t_start = time()
        my_iso = manifold.Isomap(n_components=dim)
        X_iso = my_iso.fit_transform(X)
        rd_time[line].append(time() - t_start)
        # plot_embedding(X_iso, y, 'Isomap')

        t_start = time()
        my_tsne = manifold.TSNE(n_components=dim)
        X_tsne = my_tsne.fit_transform(X)
        rd_time[line].append(time() - t_start)
        # plot_embedding(X_tsne, y, 't sne')

        t_start = time()
        my_lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=dim, method="standard")
        X_lle = my_lle.fit_transform(X)
        rd_time[line].append(time() - t_start)
        # plot_embedding(X_lle, y, 'LLE')

        t_start = time()
        my_le = manifold.SpectralEmbedding(n_components=dim, random_state=0, eigen_solver="arpack")
        X_le = my_le.fit_transform(X)
        rd_time[line].append(time() - t_start)
        # plot_embedding(X_le, y, 'Laplacian Eigenmaps')

    print(rd_time)
