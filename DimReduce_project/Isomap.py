from time import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import manifold, metrics
from sklearn.decomposition import PCA

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
    fig.colorbar(p).ax.set_yticklabels(['0: T-shirt/top', '1: Trouser', '2: Pullover', '3: Dress', '4: Coat', '5: Sandal',
                                       '6: Shirt', '7: Sneaker', '8: Bag', '9: Ankle boot'])
    plt.savefig(f"{t}_{dim}D.png")
    if plot:
        plt.show()


fashion_data = keras.datasets.fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_data
X = X_test.reshape((10000, 28*28))[:1000,:]
y = y_test[:1000]


# plot fashion MINIST
# fashion_cla = ['0: T-shirt/top', '1: Trouser', '2: Pullover', '3: Dress', '4: Coat', '5: Sandal',
#                                        '6: Shirt', '7: Sneaker', '8: Bag', '9: Ankle boot']
# fig = plt.figure()
# j = 0
# i = 0
# while i < 10:
#     if y_train[j] == i:
#         ax = fig.add_subplot(4, 3, 1+i)
#         ax.imshow(X_train[j])
#         ax.set_title(fashion_cla[y_train[j]])
#         ax.get_xaxis().set_visible(False)
#         i += 1
#     j += 1
# plt.show()

plt.set_cmap('gist_rainbow')
dim = 3

my_pca = PCA(n_components=dim)
X_pca = my_pca.fit_transform(X)
X_pca_back = my_pca.inverse_transform(X_pca)
pca_loss = np.linalg.norm(X - X_pca_back)
print(dim, "pca loss", pca_loss)
# plot_embedding(X_pca[:, 0:dim], y, 'PCA')

# my_MDS = manifold.MDS(n_components=dim, n_init=1, max_iter=100)
# X_mds = my_MDS.fit_transform(X)
# c = keras.metrics.MeanRelativeError(X_mds)
# Deuclidean = metrics.pairwise.pairwise_distances(X_mds, metric='euclidean')
# print(Deuclidean.shape)
# print(my_MDS.stress_, X_mds.shape,my_MDS.embedding_.shape,my_MDS.get_params(),)
# # mds_loss = my_MDS.stress_
# print(dim, "mds loss", mds_loss)
# plot_embedding(X_mds, y, 'MDS')
#

my_iso = manifold.Isomap(n_components=dim)
X_iso = my_iso.fit_transform(X)
iso_loss = my_iso.reconstruction_error()
print(dim, "iso loss", iso_loss)
# plot_embedding(X_iso, y, 'Isomap')

# my_tsne = manifold.TSNE(n_components=dim)
# X_tsne = my_tsne.fit_transform(X)
# plot_embedding(X_tsne, y, 't sne')
#
#
my_lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=dim, method="standard")
X_lle = my_lle.fit_transform(X)
lle_loss = my_lle.reconstruction_error_
print(dim, "lle loss", lle_loss)
# plot_embedding(X_lle, y, 'LLE')
#
#
my_le = manifold.SpectralEmbedding(n_components=dim, random_state=0, eigen_solver="arpack")
X_le = my_le.fit_transform(X)
# plot_embedding(X_le, y, 'Laplacian Eigenmaps')

