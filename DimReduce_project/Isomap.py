import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn import manifold
from sklearn.decomposition import PCA
from Utils import reduce_dim_and_nn

FASHION_CLASS = ['0: T-shirt/top', '1: Trouser', '2: Pullover', '3: Dress', '4: Coat', '5: Sandal',
                 '6: Shirt', '7: Sneaker', '8: Bag', '9: Ankle boot']
REDUCED_DIM_METHODS = [#lambda dim: PCA(n_components=dim),
                       #lambda dim: manifold.Isomap(n_components=dim),
                       # lambda dim: manifold.TSNE(n_components=dim),
                       lambda dim: manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=dim, method="standard"),
                       lambda dim: manifold.SpectralEmbedding(n_components=dim, random_state=0, eigen_solver="arpack")
                       ]



def plot_embedding(X, y, t):
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
        print("Dimension too high to plot")
        return
    ax.set_title(t)
    fig.colorbar(p).ax.set_yticklabels(FASHION_CLASS)
    plt.savefig(f"{t}_{dim}D.png")
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


def get_model_name(model):
    class_name = str(type(model))
    start, end = None, None
    for i in range(len(class_name) - 1, -1, -1):
        letter = class_name[i]
        if end is None and (letter.isalpha() or letter == "_"):
            end = i
        if letter == ".":
            start = i + 1
            return class_name[start:end + 1]


if __name__ == '__main__':
    fashion_data = keras.datasets.fashion_mnist.load_data()
    (X_train, y_train), (X_test, y_test) = fashion_data
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))


    plt.set_cmap('gist_rainbow')
    test_dim = np.array([2, 3, 5] + [10 * i for i in range(1, 11)])
    duration, accuracy = {}, {}
    model_names = []


    plot_flag = False
    for model in REDUCED_DIM_METHODS:
        model_name = get_model_name(model(2))
        model_names.append(model_name)
        duration = np.zeros((len(test_dim), 3))
        duration[:, 0] = test_dim
        accuracy = np.zeros((len(test_dim), 3))
        accuracy[:, 0] = test_dim
        for idx, dim in enumerate(test_dim):
            print("dim = ", dim, "; model = ", model_name)
            train_time, test_time, nn_train_acc, nn_test_acc = reduce_dim_and_nn(model(dim).fit_transform, dim,
                                                                                 X_train, y_train, X_test, y_test)
            duration[idx, 1], duration[idx, 2] = train_time, test_time
            accuracy[idx, 1], accuracy[idx, 2] = nn_train_acc, nn_test_acc
            with open(model_name+'_t.npy', 'wb') as f:
                np.save(f, duration)
            with open(model_name+'_acc.npy', 'wb') as f:
                np.save(f, accuracy)

