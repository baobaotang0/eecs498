from time import time
import numpy as np
import tensorflow as tf

SEGMENT_LEN = 5000


def get_dim_reduced(rd_model, dim, X):
    rd_X = np.zeros((len(X), dim))
    for i in range(len(X) // SEGMENT_LEN):
        start, end = i * SEGMENT_LEN, np.min([len(X), (i + 1) * SEGMENT_LEN])
        rd_X[start:end, :] = rd_model(X[start:end, :])
    return rd_X


def reduce_dim_and_nn(rd_model, dim, X_train, y_train, X_test, y_test):
    # fit
    t_start = time()
    rd_X_train = get_dim_reduced(rd_model, dim, X_train)
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[rd_X_train.shape[1]]),
        tf.keras.layers.Dense(16, activation=tf.nn.softmax),
        tf.keras.layers.Dense(10)])
    nn_model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
    nn_model.fit(rd_X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
    train_time = time() - t_start
    nn_train_acc = nn_model.history.history["accuracy"][-1]
    print("fit time = ", train_time, "fit accuracy = ", nn_train_acc)

    # prdict
    t_start = time()
    rd_X_test = get_dim_reduced(rd_model, dim, X_test)
    nn_test_loss, nn_test_acc = nn_model.evaluate(rd_X_test, y_test, verbose=2)
    test_time = time() - t_start
    print("predict time = ", test_time, "predict accuracy = ", nn_test_acc)
    return train_time, test_time, nn_train_acc, nn_test_acc
