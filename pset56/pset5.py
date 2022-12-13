import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt


def G(theta, d):
    return (theta**3)*(d**2)+theta*np.exp(-np.abs(0.2-d))


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


class Surrogate_NN_Model():
    def __init__(self, num_theta, num_d, EPOCHS=2000):
        self.num_theta = num_theta
        self.num_d = num_d
        self.EPOCHS = EPOCHS
        self.theta = np.linspace(0, 1, num_theta)
        self.d_set = np.linspace(0, 1, num_d)
        self.dataset = np.zeros((num_theta, num_d))
        self.train_X = np.zeros((num_theta*num_d, 2))
        self.train_y = np.zeros((num_theta*num_d, 1))
        for idx1, i in enumerate(self.theta):
            for idx2, j in enumerate(self.d_set):
                self.train_X[idx1 * num_d + idx2, :] = [i, j]
                self.train_y[idx1 * num_d + idx2] = G(i, j)
                self.dataset[idx1, idx2] = G(i, j)

        self.get_nn()

    def get_nn(self):
        self.model = keras.Sequential([
            keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[2]),
            keras.layers.Dense(64, activation=tf.nn.softmax),
            keras.layers.Dense(1)])
        self.model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        history = self.model.fit(self.train_X, self.train_y, epochs=self.EPOCHS, validation_split=0.2, verbose=0,
                                      callbacks=[early_stop, PrintDot()])
        self.hist = pd.DataFrame(history.history)
        self.hist['epoch'] = history.epoch


def norm_pdf(x, loc=0.0, scale=1.0):
    return np.exp(-(x-loc)**2/2/scale**2)/scale/np.sqrt(2*np.pi)


def norm_logpdf(x, loc=0.0, scale=1.0):
    return -np.log(np.sqrt(2 * np.pi) * scale) - (x - loc) ** 2 / 2 / scale ** 2


def get_utility(num_d, N1, N2, model_fun=G, sigma=0.01):
    d_set = np.linspace(0, 1, num_d)
    U = np.zeros((num_d,))
    theta_i = np.random.uniform(size=(N1, 1))
    theta_ij = np.random.uniform(size=(N2, 1))
    noise = np.random.normal(size=(N1, 1))
    for k, d in enumerate(d_set):
        G_i, G_ij = model_fun(theta_i, d), model_fun(theta_ij, d)
        ys = G_i + sigma * noise
        log_likelihood = norm_logpdf(ys, G_i, sigma)
        evidence = np.array([np.mean(norm_pdf(ys[i:i + 1], G_ij, sigma)) for i in range(N1)])
        U[k] = np.mean(log_likelihood - np.log(evidence))
    return d_set, U


if __name__ == '__main__':
    SNM = Surrogate_NN_Model(200, 200)

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('value loss')
    plt.plot(SNM.hist['epoch'], SNM.hist['loss'],
             label='Train Error')
    plt.plot(SNM.hist['epoch'], SNM.hist['val_loss'],
             label='Val Error')
    plt.legend()
    plt.show()

    new_y = SNM.model.predict(SNM.train_X).reshape(SNM.dataset.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(SNM.theta, SNM.d_set)
    ax.plot_surface(X, Y, SNM.dataset.T, color="r")
    ax.plot_surface(X, Y, new_y.T, color="g")
    plt.show()

    nn = lambda theta, d: SNM.model.predict(np.concatenate((theta, np.ones((len(theta), 1))*d), axis=1))
    d_set, U = get_utility(50, 1000, 1000, nn)
    plt.plot(d_set, U)
    plt.show()




