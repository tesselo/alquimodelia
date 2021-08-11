import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv3D,
    Conv2D,
    Conv3DTranspose,
    Dense,
    Dropout,
    Flatten,
    GlobalMaxPool3D,
    Input,
    MaxPooling3D,
    Reshape,
    add,
    concatenate,
    Permute,
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, plot_model, to_categorical


def conv3d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, data_format='channels_first', padding='same', activation="relu",):
    # first layer
    x = Conv3D(
        filters=n_filters,
        kernel_size=kernel_size,
        kernel_initializer="he_normal",
        padding=padding,
        data_format=data_format,
        activation=activation,
    )(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    # Second layer.
    x = Conv3D(
        filters=n_filters,
        kernel_size=kernel_size,
        kernel_initializer="he_normal",
        padding=padding,
        activation=activation,
        data_format=data_format,
    )(x)
    if batchnorm:
        x = BatchNormalization()(x)
    return x


def contracting_block(
    input_img, n_filters, batchnorm, dropout=0.5, kernel_size=3, strides=2, data_format='channels_last', padding='same', activation="relu"
):
    c1 = conv3d_block(
        input_img, n_filters=n_filters * 1, kernel_size=kernel_size, batchnorm=batchnorm, data_format=data_format, activation=activation, padding=padding
    )
    p1 = MaxPooling3D(strides, padding=padding)(c1)
    p1 = Dropout(dropout * 0.5)(p1)
    return p1, c1


def expansive_block(
    ci, cii, n_filters, batchnorm, dropout=0.5, kernel_size=3, strides=2, data_format='channels_first', activation="relu", padding="same",
):
    u = Conv3DTranspose(n_filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format)(ci)
    u = concatenate([u, cii])
    u = Dropout(dropout)(u)
    c = conv3d_block(u, n_filters=n_filters, kernel_size=kernel_size, batchnorm=batchnorm, data_format=data_format, activation=activation, padding=padding)
    return c


def get_unet_12t(input_img, n_filters=16, dropout=0.5, batchnorm=True, data_format='channels_last', activation_end="sigmoid", activation_middle="relu", kernel_size=3, padding="same", local=False, num_classes=1):
    # contracting path
    p1, c1 = contracting_block(
        input_img, n_filters, batchnorm, dropout=0.5, kernel_size=kernel_size, data_format=data_format, activation=activation_middle,padding=padding,
    )
    p2, c2 = contracting_block(p1, n_filters * 2, batchnorm, dropout=0.5, kernel_size=kernel_size, data_format=data_format, activation=activation_middle, padding=padding)

    # middle
    p3, c3 = contracting_block(p2, n_filters * 4, batchnorm, dropout=0.5, kernel_size=kernel_size, data_format=data_format, activation=activation_middle, padding=padding)


    # expansive path
    c4 = expansive_block(c3, c2, n_filters * 2, batchnorm, dropout=0.5, data_format=data_format, activation=activation_middle, kernel_size=kernel_size)
    c5 = expansive_block(c4, c1, n_filters * 1, batchnorm, dropout=0.5, data_format=data_format, activation=activation_middle, kernel_size=kernel_size)
    outputs = Conv3D(1, 1, activation=activation_end, data_format=data_format, padding=padding)(c5)
    width = outputs.shape[2]
    height = outputs.shape[3]
    final_shape = (width, height, num_classes)
    #squeeze_shape = (outputs.shape[1:-1])
    #print(squeeze_shape)
    #print(type(squeeze_shape))
    #outputs = Reshape(squeeze_shape)(outputs)

    # To be able to run on CPU intead of a inverse conv3d, we need to change the axis
    if local == True:
        re_shape = (outputs.shape[-1], *outputs.shape[2:-1], outputs.shape[1])
        print(outputs.shape)
        print('re', re_shape)
        outputs = Reshape(re_shape)(outputs)
        print(outputs.shape)
        outputs2 = Conv3D(num_classes, 1, activation=activation_end, data_format=data_format, padding=padding)(outputs)
        print(outputs.shape)

    else:
        outputs2 = Conv3D(num_classes, 1, activation=activation_end, data_format='channels_first', padding=padding)(outputs)
    print('bef',outputs2.shape)
    print('want',final_shape)
    outputs2 = Reshape(final_shape)(outputs2)
    print(outputs2.shape)

    return outputs2

def get_unet_12t_2stps(input_img, n_filters=16, dropout=0.5, batchnorm=True, data_format='channels_last', activation_end="sigmoid", activation_middle="relu", kernel_size=3, padding="same", local=False, num_classes=1):
    # contracting path
    p1, c1 = contracting_block(
        input_img, n_filters, batchnorm, dropout=0.5, kernel_size=kernel_size, data_format=data_format, activation=activation_middle,padding=padding,
    )

    p2, c2 = contracting_block(p1, n_filters * 2, batchnorm, dropout=0.5, kernel_size=kernel_size, data_format=data_format, activation=activation_middle, padding=padding)

    # middle
    #p3, c3 = contracting_block(p2, n_filters * 4, batchnorm, dropout=0.5, data_format=data_format)


    # expansive path
    c3 = expansive_block(c2, c1,n_filters * 2, batchnorm, dropout=0.5, data_format=data_format, activation=activation_middle, kernel_size=kernel_size)
    #c4 = expansive_block(c3, c2, n_filters * 1, batchnorm, dropout=0.5, kernel_size=3, data_format=data_format)
    outputs = Conv3D(1, 1, activation=activation_end, data_format=data_format, padding=padding)(c3)
    print(outputs.shape)
    width = outputs.shape[2]
    height = outputs.shape[3]
    final_shape = (num_classes, width, height)

    # To be able to run on CPU intead of a inverse conv3d, we need to change the axis
    if local == True:
        re_shape = (outputs.shape[-1], *outputs.shape[2:-1], outputs.shape[1])
        outputs = Reshape(re_shape)(outputs)
        outputs2 = Conv3D(1, 1, activation=activation_end, data_format=data_format, padding=padding)(outputs)
    else:
        outputs2 = Conv3D(num_classes, 1, activation=activation_end, data_format='channels_first', padding=padding)(outputs)
    outputs2 = Reshape(final_shape)(outputs2)
    return outputs2


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

def nan_mean_square_error_loss(nan_value=np.nan):
    # Create a loss function
    def loss(y_true, y_pred):
        indices = tf.where(
            tf.not_equal(y_true, nan_value)
        )  #  or `tf.less`, `tf.equal` etc.
        return tf.keras.losses.mean_square_error(
            tf.gather(y_true, indices), tf.gather(y_pred, indices)
        )

    # Return a function
    return loss
