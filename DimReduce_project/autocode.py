from keras.layers import Input, Dense
from keras.models import Model
from tensorflow import keras
import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np
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
    autoencoder.fit(x_train, x_train, epochs=epoch, batch_size=BATCH_SIZE, shuffle=True, callbacks=None)

    return encoder, autoencoder


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    dim = 2

    encoder, autoencoder = train(x_train=x_train, dim=2)
    encode_images = encoder.predict(x_train)


    nn_model = keras.Sequential([
        keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[dim]),
        keras.layers.Dense(16, activation=tf.nn.softmax),
        keras.layers.Dense(10)])
    nn_model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    nn_model.fit(encode_images, y_train, epochs=10, validation_split=0.2, verbose=0)

    nn_train_acc = nn_model.history.history["accuracy"][-1]
    nn_epoch = len(nn_model.history.history["accuracy"])
    print("epoch = ", nn_epoch, "accuracy = ", nn_train_acc)

    nn_test_loss, nn_test_acc = nn_model.evaluate(encoder.predict(x_test), y_test, verbose=2)
    print("test accuracy = ", nn_test_acc)


