import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from matplotlib import pyplot as plt


def G(theta, d):
    return theta**3*d**2+theta*np.exp(-np.abs(0.2-d))

def norm(x):
  return (x - np.mean(x)) / np.std(x)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

if __name__ == '__main__':
    #Begin by generating a training dataset using the G function above
    num_theta, num_d = 100, 500
    theta = np.linspace(0, 1, num_theta) #np.random.uniform(size=num_theta)
    d = np.linspace(0, 1, num_d)
    dataset = np.zeros((num_theta, num_d))
    trX = np.zeros((num_theta*num_d, 2))
    trY = np.zeros((num_theta*num_d, 1))
    for idx1, i in enumerate(theta):
        for idx2, j in enumerate(d):
            trX[idx1*num_d+idx2, :] = [i, j]
            trY[idx1*num_d+idx2] = G(i, j)
            dataset[idx1, idx2] = G(i, j)


    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[2]),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    model.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    EPOCHS = 50
    history = model.fit(trX, trY, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    # plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    # plt.ylim([0, 20])
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('MOodel')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    # plt.ylim([0, 20])
    plt.legend()
    plt.show()


    # new_y = model.predict(trX)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # X, Y = np.meshgrid(theta, d)
    # ax.plot_wireframe(X, Y, dataset.T)
    # plt.show()


    # plt.plot(trY,"g")
    # plt.plot(model.predict(trX))
    # plt.show()
    # print()


    # X = tf.placeholder("float", [None, 2])
    # Y = tf.placeholder("float", [None, 1])
    #
    # w_h = tf.Variable(tf.random_normal([2, 16], stddev=0.01))
    # w_o = tf.Variable(tf.random_normal([16, 1], stddev=0.01))
    #
    # py_x = tf.matmul(tf.nn.sigmoid(tf.matmul(X, w_h)), w_o)(X, w_h, w_o)
    #
    # predict_op = tf.argmax(py_x,1)
    #
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    # train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    #
    # with tf.Session() as sess:
    #     tf.initialize_all_variables().run()
    #
    #     for i in range(1000):
    #         for start, end in zip(range(0, len(trX), 128), range(128, len(trX) + 1, 128)):
    #             sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    #         # accu = np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX}))
    #
    #



