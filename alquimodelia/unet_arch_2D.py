import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    GlobalMaxPool2D,
    Input,
    MaxPooling2D,
    Reshape,
    add,
    concatenate,
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, plot_model, to_categorical


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        kernel_initializer="he_normal",
        padding="same",
        data_format="channels_last",
    )(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        kernel_initializer="he_normal",
        padding="same",
    )(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def contracting_block(
    input_img, n_filters, batchnorm, dropout=0.5, kernel_size=3, strides=2
):
    c1 = conv2d_block(
        input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    p1 = MaxPooling2D(strides, padding="same")(c1)
    p1 = Dropout(dropout * 0.5)(p1)
    return p1, c1


def expansive_block(
    ci, cii, n_filters, batchnorm, dropout=0.5, kernel_size=3, strides=2
):

    u = Conv2DTranspose(n_filters, 3, strides=strides, padding="same")(ci)
    u = concatenate([u, cii])
    u = Dropout(dropout)(u)
    c = conv2d_block(u, n_filters=n_filters, kernel_size=3, batchnorm=batchnorm)
    return c


def get_unet_10b_2D(input_img, n_filters=16, dropout=0.5, batchnorm=True, num_classes=1):
    # contracting path
    p1, c1 = contracting_block(
        input_img, n_filters, batchnorm, dropout=0.5, kernel_size=3
    )
    p2, c2 = contracting_block(p1, n_filters * 2, batchnorm, dropout=0.5, kernel_size=3)
    # middle
    p3, c3 = contracting_block(p2, n_filters * 4, batchnorm, dropout=0.5)
    # expansive path
    c4 = expansive_block(c3, c2, n_filters * 2, batchnorm, dropout=0.5, kernel_size=3)
    c5 = expansive_block(c4, c1, n_filters * 1, batchnorm, dropout=0.5, kernel_size=3)
    outputs = Conv2D(num_classes, 1, activation="sigmoid")(c5)
    # outputs2 = Conv@D(1, 1, activation="sigmoid", data_format="channels_first")(outputs)
    return outputs


def nan_binary_crossentropy_loss(nan_value=np.nan):
    # Create a loss function
    def loss(y_true, y_pred):
        indices = tf.where(
            tf.not_equal(y_true, nan_value)
        )  #  or `tf.less`, `tf.equal` etc.
        return tf.keras.losses.binary_crossentropy(
            tf.gather(y_true, indices), tf.gather(y_pred, indices)
        )

    # Return a function
    return loss
