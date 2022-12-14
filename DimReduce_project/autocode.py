from keras.layers import Input, Dense
from keras.models import Model
from tensorflow import keras
import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np
from Utils import reduce_dim_and_nn
import matplotlib.pyplot as plt

ENCODING_DIM_INPUT = 784
BATCH_SIZE = 64

def train(x_train, dim, epoch=50):
    """
    build autoencoder.
    :param x_train:  the train data
    :return: encoder and decoder
    """
    # input placeholder
    input_image = Input(shape=(ENCODING_DIM_INPUT, ))

    # encoding layer
    hidden_layer = Dense(dim, activation='relu')(input_image)
    # decoding layer
    decode_output = Dense(ENCODING_DIM_INPUT, activation='relu')(hidden_layer)

    # build autoencoder, encoder, decoder
    autoencoder = Model(inputs=input_image, outputs=decode_output)
    encoder = Model(inputs=input_image, outputs=hidden_layer)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    autoencoder.fit(x_train, x_train, epochs=epoch, batch_size=BATCH_SIZE, shuffle=True, callbacks=[early_stop])

    return encoder, autoencoder


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    test_dim = np.array([2, 3, 5] + [10 * i for i in range(1, 11)])

    model_names = ["autoencoder"]
    for model_name in model_names:
        duration = np.zeros((len(test_dim), 3))
        duration[:, 0] = test_dim
        accuracy = np.zeros((len(test_dim), 3))
        accuracy[:, 0] = test_dim
        for idx, dim in enumerate(test_dim):
            print("dim = ", dim, "; model = ", model_name)
            encoder, autoencoder = train(x_train=X_train, dim=dim)
            train_time, test_time, nn_train_acc, nn_test_acc = reduce_dim_and_nn(encoder.predict, dim,
                                                                                 X_train, y_train, X_test, y_test)
            duration[idx, 1], duration[idx, 2] = train_time, test_time
            accuracy[idx, 1], accuracy[idx, 2] = nn_train_acc, nn_test_acc
        with open(model_name+'_t.npy', 'wb') as f:
            np.save(f, duration)
        with open(model_name+'_acc.npy', 'wb') as f:
            np.save(f, accuracy)

