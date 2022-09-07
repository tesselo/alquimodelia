from functools import cached_property
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    AveragePooling3D,
    BatchNormalization,
    Conv2D,
    Conv3D,
    Dense,
    Flatten,
    MaxPooling2D,
    MaxPooling3D,
    UpSampling2D,
    UpSampling3D,
    ZeroPadding2D,
    ZeroPadding3D,
    add,
)
from tensorflow.keras.regularizers import l2

from alquimodelia.alquimodelia import ModelMagia

# based on https://towardsdatascience.com/implementing-a-resnet-model-from-scratch-971be7193718


class ResNet(ModelMagia):
    """Base classe for ResNet models"""

    def __init__(
        self,
        stages: Tuple[int, int, int, int] = (1, 2, 3, 4),
        n_filters: int = 16,
        reg: float = 0.0001,
        bnEps: float = 2e-5,
        bnMom: float = 0.9,
        upsample: int = 0,
        padding_style: str = "same",
        activation_middle: str = "relu",
        **kwargs,
    ):
        self.stages = stages
        self.n_filters = n_filters
        self.upsample = upsample
        self.reg = reg
        self.bnEps = bnEps
        self.bnMom = bnMom
        self.padding_style = padding_style
        self.activation_middle = activation_middle
        self.filters_list = [2]
        for i in range(1, len(self.stages) + 1):
            self.filters_list.append(2 ** (i + 2))
        super().__init__(**kwargs)

    def residual_module(
        self,
        data,
        filter,
        stride,
        chanDim,
        red=False,
        reg=0.0001,
        bnEps=2e-5,
        bnMom=0.9,
    ):
        """
        ResNet convolution module.

        Parameters
        ----------
            data : keras.layer
                Layer with the input image.
            filter : int
                Filters to aply.
            stride : int
                Strides of the convolution along each spatial dimension.
            chanDim : int
                Axis to aply normalization.
            red : Bool
                Deciding to reduce Spatial dimension.
            reg : float
                Regulizer factor.
            bnEps : float
                Small float added to variance to avoid dividing by zero
            bnMom : float
                Momentum for the moving average.
        Returns
        ----------
            x : keras.layer
                Last layer of Module
        """

        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation(self.activation_middle)(bn1)
        conv1 = self.Conv(
            int(filter * 0.25), 1, use_bias=False, kernel_regularizer=l2(reg)
        )(act1)
        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation(self.activation_middle)(bn2)
        conv2 = self.Conv(
            int(filter * 0.25),
            3,
            strides=stride,
            padding=self.padding_style,
            use_bias=False,
            kernel_regularizer=l2(reg),
        )(act2)
        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation(self.activation_middle)(bn3)
        conv3 = self.Conv(filter, 1, use_bias=False, kernel_regularizer=l2(reg))(act3)
        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = self.Conv(
                filter, 1, strides=stride, use_bias=False, kernel_regularizer=l2(reg)
            )(act1)
        x = add([conv3, shortcut])
        return x

    def output_layer(self) -> tf.Layer:
        """
        Build the Model architeture and return last layer.

        Returns
        ----------
            x : keras.layer
                Last layer of ResNet deep neural network
        """
        chanDim = -1

        if self.data_format == "channels_first":
            chanDim = 2
        # set the input and apply BN
        inputs = self.input_layer
        if self.upsample != 0:
            x = self.UpSampling(size=(1, self.upsample, self.upsample))(inputs)
        else:
            x = inputs
        x = BatchNormalization(axis=chanDim, epsilon=self.bnEps, momentum=self.bnMom)(x)

        # apply CONV => BN => ACT => POOL to reduce spatial size
        x = self.Conv(
            self.filters_list[0],
            5,
            use_bias=False,
            padding=self.padding_style,
            kernel_regularizer=l2(self.reg),
        )(x)
        x = BatchNormalization(axis=chanDim, epsilon=self.bnEps, momentum=self.bnMom)(x)
        x = Activation(self.activation_middle)(x)
        x = self.ZeroPadding(1)(x)
        x = self.MaxPooling(3, strides=2)(x)
        # loop over the number of stages
        for i in range(0, len(self.stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = 1 if i == 0 else 2
            x = self.residual_module(
                x,
                self.filters_list[i + 1],
                stride,
                chanDim,
                red=True,
                bnEps=self.bnEps,
                bnMom=self.bnMom,
            )

            # loop over the number of layers in the stage
            for j in range(0, self.stages[i] - 1):
                x = self.residual_module(
                    x,
                    self.filters_list[i + 1],
                    1,
                    chanDim,
                    bnEps=self.bnEps,
                    bnMom=self.bnMom,
                )
        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=self.bnEps, momentum=self.bnMom)(x)
        x = Activation(self.activation_middle)(x)
        min_value_inshape = min([f for f in x.shape if f is not None])
        x = self.AveragePooling(min(6, min_value_inshape))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(self.num_classes, kernel_regularizer=l2(self.reg))(x)
        x = Activation(self.activation_final)(x)
        return x


class ResNet2D(ResNet):
    def __init__(
        self,
        **kwargs,
    ):
        self.Conv = Conv2D
        self.ZeroPadding = ZeroPadding2D
        self.MaxPooling = MaxPooling2D
        self.AveragePooling = AveragePooling2D
        self.UpSampling = UpSampling2D
        kwargs["timesteps"] = 1

        super().__init__(**kwargs)

    @cached_property
    def model_input_shape(self):
        """Defines input shape based on channel dimension"""
        if self.data_format == "channels_first":
            return (self.num_bands, self.height, self.width)
        elif self.data_format == "channels_last":
            return (self.height, self.width, self.num_bands)


class ResNet3D(ResNet):
    def __init__(
        self,
        **kwargs,
    ):
        self.Conv = Conv3D
        self.ZeroPadding = ZeroPadding3D
        self.MaxPooling = MaxPooling3D
        self.AveragePooling = AveragePooling3D
        self.UpSampling = UpSampling3D

        super().__init__(**kwargs)

    @cached_property
    def model_input_shape(self):
        """Defines input shape based on channel dimension"""
        if self.data_format == "channels_first":
            return (
                self.timesteps,
                self.num_bands,
                self.height,
                self.width,
            )
        elif self.data_format == "channels_last":
            return (
                self.timesteps,
                self.height,
                self.width,
                self.num_bands,
            )
