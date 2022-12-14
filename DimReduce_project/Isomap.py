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
REDUCED_DIM_METHODS = [lambda dim: PCA(n_components=dim),
                       lambda dim: manifold.Isomap(n_components=dim),
                       # lambda dim: manifold.TSNE(n_components=dim),
                       lambda dim: manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=dim, method="standard"),
                       lambda dim: manifold.SpectralEmbedding(n_components=dim, random_state=0, eigen_solver="arpack")
                       ]
SEGMENT_LEN = 5000


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
    for i in range(len(class_name) - 1, -1, -1):
        letter = class_name[i]
        if end is None and (letter.isalpha() or letter == "_"):
            end = i
        if letter == ".":
            start = i + 1
            return class_name[start:end + 1]


class ReducedDimNN:
    def __init__(self, rd_model, X, y):
        self.rd_model = rd_model
        self.dim = self.rd_model.get_params()['n_components']
        self.rd_X = np.zeros((len(X), self.dim))
        t_start = time()
        for i in range(len(X) // SEGMENT_LEN):
            start, end = i * SEGMENT_LEN, np.min([len(X), (i + 1) * SEGMENT_LEN])
            print(start, end)
            self.rd_X[start:end, :] = self.rd_model.fit_transform(X[start:end, :])
        self.rd_time = time() - t_start
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
        t_start = time()
        self.nn_model.fit(self.rd_X, self.labels, epochs=20, validation_split=0.2, verbose=0)
        self.nn_train_time = time() - t_start
        self.nn_train_acc = self.nn_model.history.history["accuracy"][-1]
        self.nn_epoch = len(self.nn_model.history.history["accuracy"])
        print("epoch = ", self.nn_epoch, "accuracy = ", self.nn_train_acc)

    def predict(self, X_test, y_test):
        t_start = time()
        self.nn_test_loss, self.nn_test_acc = self.nn_model.evaluate(self.rd_model.fit_transform(X_test), y_test,
                                                                     verbose=2)
        self.nn_test_time = time() - t_start
        print("predict time = ", self.nn_test_time, "accuracy = ", self.nn_test_acc)


if __name__ == '__main__':
    fashion_data = keras.datasets.fashion_mnist.load_data()
    (X_train, y_train), (X_test, y_test) = fashion_data
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    y = y_train
    X_te = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    y_te = y_test

    plt.set_cmap('gist_rainbow')
    test_dim = [2] + [10 * i for i in range(1, 11)]
    models, duration, accuracy = {}, {}, {}
    model_names = []

    plot_flag = False
    for dim in test_dim:
        for model in [  # PCA(n_components=dim),
            manifold.Isomap(n_components=dim),
            # manifold.TSNE(n_components=dim),
            manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=dim, method="standard"),
            manifold.SpectralEmbedding(n_components=dim, random_state=0, eigen_solver="arpack")
        ]:
            model_name = get_model_name(model)
            print("dim = ", dim, "; model = ", model_name)
            if model_name not in duration.keys():
                model_names.append(model_name)
                models[model_name] = []
                duration[model_name] = []
                accuracy[model_name] = []
            local_model = ReducedDimNN(model, X, y)
            local_model.get_nn()
            local_model.predict(X_te, y_te)
            models[model_name].append(local_model)
            duration[model_name].append([local_model.rd_time, local_model.nn_train_time, local_model.nn_test_time])
            accuracy[model_name].append([local_model.nn_train_acc, local_model.nn_test_acc])

    duration = np.array(duration)
    accuracy = np.array(accuracy)
    import json
    import datetime
    import numpy as np


    class JsonEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.__str__()
            else:
                return super(JsonEncoder, self).default(obj)


    with open('t.json', 'w') as json_file:
        json.dump(duration, json_file, ensure_ascii=False, cls=JsonEncoder)
    with open('acc.json', 'w') as json_file:
        json.dump(accuracy, json_file, ensure_ascii=False, cls=JsonEncoder)

    fig_t = plt.figure()
    ax_rd_t, ax_tr_t, ax_tol_t, ax_te_t = fig_t.add_subplot(2, 2, 1), fig_t.add_subplot(2, 2, 2), fig_t.add_subplot(2,
                                                                                                                    2,
                                                                                                                    3), fig_t.add_subplot(
        2, 2, 4)
    fig_acc = plt.figure()
    ax_tr_acc, ax_te_acc = fig_acc.add_subplot(2, 1, 1), fig_acc.add_subplot(2, 1, 2)
    for model_name in model_names:
        ax_rd_t.plot(test_dim, duration[model_name][:, 0], label=model_name)
        ax_tr_t.plot(test_dim, duration[model_name][:, 1], label=model_name)
        ax_tol_t.plot(test_dim, duration[model_name][:, 0] + duration[model_name][:, 1], label=model_name)
        ax_te_t.plot(test_dim, duration[model_name][:, 2], label=model_name)
        ax_te_acc.plot(test_dim, accuracy[model_name][:, 0], label=model_name)
        ax_te_acc.plot(test_dim, accuracy[model_name][:, 1], label=model_name)

    ax_rd_t.add_title("duration of dimension reduction")
    ax_tr_t.add_title("duration of training nn")
    ax_tol_t.add_title("duration of building model")
    ax_te_t.add_title("duration of predicting")
    ax_tr_acc.add_title("accuracy of training")
    ax_te_acc.add_title("accuracy of testing")
    plt.show()
    # print(res)
