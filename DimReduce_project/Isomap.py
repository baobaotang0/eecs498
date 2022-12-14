from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import manifold
from sklearn.decomposition import PCA

FASHION_CLASS = ['0: T-shirt/top', '1: Trouser', '2: Pullover', '3: Dress', '4: Coat', '5: Sandal',
                 '6: Shirt', '7: Sneaker', '8: Bag', '9: Ankle boot']


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
    for i in range(len(class_name)-1,-1,-1):
        letter = class_name[i]
        if end is None and (letter.isalpha() or letter == "_"):
            end = i
        if letter == ".":
            start = i+1
            return class_name[start:end+1]


class ReducedDimNN:
    def __init__(self, rd_model, X, y):
        t_start = time()
        self.rd_model = rd_model
        self.rd_X = self.rd_model.fit_transform(X)
        self.rd_time = time() - t_start
        self.dim = self.rd_model.get_params()['n_components']
        self.rd_model_name = get_model_name(self.rd_model)
        self.labels = y

    def plot_reduce_dim(self):
        if self.dim == 2 and self.dim == 3:
            if self.rd_model_name == "PCA":
                plot_embedding(self.rd_X[:, 0:dim], self.labels, self.rd_model_name)
            else:
                plot_embedding(self.rd_X, self.labels, self.rd_model_name)
        else:
            print("Dimension too high to plot")

    def get_nn(self):
        self.nn_model = keras.Sequential([
            keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[self.dim]),
            keras.layers.Dense(16, activation=tf.nn.softmax),
            keras.layers.Dense(10)])
        self.nn_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.nn_model.fit(self.rd_X, self.labels, epochs=1000, validation_split=0.2, verbose=0,
                                      callbacks=[early_stop])
        self.nn_train_acc = self.nn_model.history.history["accuracy"][-1]
        print("accuracy = ",self.nn_train_acc)


if __name__ == '__main__':
    fashion_data = keras.datasets.fashion_mnist.load_data()
    (X_train, y_train), (X_test, y_test) = fashion_data
    X = X_test.reshape((10000, 28 * 28))[:3000, :]
    y = y_test[:3000]

    plt.set_cmap('gist_rainbow')
    test_dim = [2, 10, 20, 50, 100]
    res = {}
    plot_flag = False
    for dim in test_dim:
        res[dim] = {}

        for model in [PCA(n_components=dim),
                      manifold.Isomap(n_components=dim),
                      #manifold.TSNE(n_components=dim),
                      manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=dim, method="standard"),
                      manifold.SpectralEmbedding(n_components=dim, random_state=0, eigen_solver="arpack")
                      ]:
            model_name = get_model_name(model)
            print("dim = ", dim, "; model = ",  model_name)
            res[dim][model_name] = ReducedDimNN(model, X, y)
            res[dim][model_name].get_nn()
            if plot_flag:
                res[-1][-1].plot_reduce_dim()



    # print(res)
