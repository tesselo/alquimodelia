from functools import cached_property
from typing import Any, Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    Cropping2D,
    MaxPooling2D,
    MaxPooling3D,
    SpatialDropout2D,
    SpatialDropout3D,
    concatenate,
)

from alquimodelia.alquimodelia import ModelMagia

# based on https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47


def count_number_divisions(size: int, count: int, by: int = 2, limit: int = 2):
    """
    Count the number of possible steps.

    Parameters
    ----------
        size : int
            Image size (considering it is a square).
        count : int
            Input must be 0.
        by : int
        limit : int
            Size of last filter (smaller).
    """
    if size >= limit:
        if size % 2 == 0:
            count = count_number_divisions(size / by, count + 1, by=by, limit=limit)
    else:
        count = count - 1
    return count


BatchNormalization_args = {}
Activation_args = {}
MaxPooling_args = {}
Dropout_args = {}
Conv_args = {}


class UNet(ModelMagia):
    """Base classe for Unet models"""

    def __init__(
        self,
        n_filters: int = 16,
        number_of_conv_layers: int = 0,
        kernel_size: int = 3,
        batchnorm: bool = True,
        padding_style: str = "same",
        padding: int = 0,
        activation: str = "relu",
        kernel_initializer: str = "he_normal",
        **kwargs,
    ):
        self._number_of_conv_layers = number_of_conv_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batchnorm = batchnorm
        self.padding = padding
        self.padding_style = padding_style

        self.activation = activation
        self.kernel_initializer = kernel_initializer
        super().__init__(**kwargs)

    @cached_property
    def number_of_conv_layers(self):
        if self._number_of_conv_layers != 0:
            number_of_layers = []
            if self.data_format == "channels_first":
                study_shape = self.model_input_shape[1:]
            elif self.data_format == "channels_last":
                study_shape = self.model_input_shape[:-1]
            for size in study_shape:
                number_of_layers.append(count_number_divisions(size, 0))

            self._number_of_conv_layers = min(number_of_layers)

        return self._number_of_conv_layers

    def inside_upsampling(self, inputs, channels, upscale_factor):

        conv_args = {
            "activation": "relu",
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        x = self.Conv(64, 5, **conv_args)(inputs)
        x = self.Conv(64, 3, **conv_args)(x)
        x = self.Conv(32, 3, **conv_args)(x)
        x = self.Conv(channels * (upscale_factor**2), 3, **conv_args)(x)
        if len(x.shape) > 4:
            t = [f for f in tf.split(x, x.shape[1], axis=1)]
            outputs = []
            for r in t:
                print(r.shape)
                outputs.append(
                    tf.nn.depth_to_space(
                        tf.keras.backend.squeeze(r, 1),
                        upscale_factor,
                        data_format="NHWC",
                    )
                )
            outputs = tf.stack(outputs, axis=1, name="stack")
        else:
            outputs = tf.nn.depth_to_space(x, upscale_factor)

        return outputs

    def convolution_block(
        self,
        input_tensor: tf.Layer,
        n_filters: int,
        kernel_size: int = 3,
        batchnorm: bool = True,
        data_format: str = "channels_first",
        padding: str = "same",
        activation: str = "relu",
        kernel_initializer: str = "he_normal",
    ) -> tf.Layer:
        # first layer
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        # Second layer.
        x = self.Conv(
            filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            padding=padding,
            data_format=data_format,
            activation=activation,
        )(x)
        if batchnorm:
            x = BatchNormalization()(x)
        return x

    def contracting_block(
        self,
        input_img: tf.Layer,
        n_filters: int = 16,
        batchnorm: bool = True,
        dropout: float = 0.25,
        kernel_size: int = 3,
        strides: int = 2,
        data_format: str = "channels_last",
        padding_style: str = "same",
        activation: str = "relu",
    ) -> Tuple[tf.Layer, tf.Layer]:
        c1 = self.convolution_block(
            input_img,
            n_filters=n_filters,
            kernel_size=kernel_size,
            batchnorm=batchnorm,
            data_format=data_format,
            activation=activation,
            padding=padding_style,
        )
        p1 = self.MaxPooling(strides, padding=padding_style)(c1)
        p1 = self.SpatialDropout(dropout, data_format=data_format)(p1)
        return p1, c1

    def expansive_block(
        self,
        ci: tf.Layer,
        cii: tf.Layer,
        n_filters: int = 16,
        batchnorm: bool = True,
        dropout: float = 0.5,
        kernel_size: int = 3,
        strides: int = 2,
        data_format: str = "channels_first",
        activation: str = "relu",
        padding_style: str = "same",
    ) -> Tuple[tf.Layer]:
        u = self.ConvTranspose(
            n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding_style,
            data_format=data_format,
        )(ci)
        u = concatenate([u, cii])
        u = self.SpatialDropout(dropout, data_format=data_format)(u)
        c = self.convolution_block(
            u,
            n_filters=n_filters,
            kernel_size=kernel_size,
            batchnorm=batchnorm,
            data_format=data_format,
            activation=activation,
            padding=padding_style,
        )
        return c

    def contracting_loop(
        self, input_img: tf.Layer, contracting_arguments: Dict[str, Any]
    ) -> List[tf.Layer]:
        list_p = [input_img]
        list_c = []
        n_filters = contracting_arguments["n_filters"]
        for i in range(self.number_of_conv_layers + 1):
            old_p = list_p[i]
            filter_expansion = 2**i
            contracting_arguments["n_filters"] = n_filters * filter_expansion
            p, c = self.contracting_block(old_p, **contracting_arguments)
            list_p.append(p)
            list_c.append(c)
        return list_c

    def expanding_loop(
        self, contracted_layers: List[tf.Layer], expansion_arguments: Dict[str, Any]
    ) -> tf.Layer:
        list_c = [contracted_layers[-1]]
        iterator_expanded_blocks = range(self.number_of_conv_layers)
        iterator_contracted_blocks = reversed(iterator_expanded_blocks)
        n_filters = expansion_arguments["n_filters"]
        for i, c in zip(iterator_expanded_blocks, iterator_contracted_blocks):
            filter_expansion = 2 ** (c)
            expansion_arguments["n_filters"] = n_filters * filter_expansion
            c4 = self.expansive_block(
                list_c[i], contracted_layers[c], **expansion_arguments
            )
            list_c.append(c4)
        return c4

    def deep_neural_network(
        self,
        n_filters: int = 16,
        dropout: float = 0.2,
        batchnorm: bool = True,
        data_format: str = "channels_last",
        activation_middle: str = "relu",
        kernel_size: int = 3,
        padding: str = "same",
    ) -> tf.Layer:
        """Build deep neural network."""

        input_img = self.input_layer
        self.define_number_convolution_layers()

        contracting_arguments = {
            "n_filters": n_filters,
            "batchnorm": batchnorm,
            "dropout": dropout,
            "kernel_size": kernel_size,
            "padding": padding,
            "data_format": data_format,
            "activation": activation_middle,
        }
        expansion_arguments = {
            "n_filters": n_filters,
            "batchnorm": batchnorm,
            "dropout": dropout,
            "data_format": data_format,
            "activation": activation_middle,
            "kernel_size": kernel_size,
        }

        contracted_layers = self.contracting_loop(input_img, contracting_arguments)
        unet_output = self.expanding_loop(contracted_layers, expansion_arguments)

        return unet_output


class UNet2D(UNet):
    def __init__(
        self,
        **kwargs,
    ):
        self.Conv = Conv2D
        self.ConvTranspose = Conv2DTranspose
        self.SpatialDropout = SpatialDropout2D
        self.MaxPooling = MaxPooling2D
        kwargs["timesteps"] = 1

        super().__init__(**kwargs)

    @cached_property
    def model_input_shape(self):
        if self.data_format == "channels_first":
            return (self.num_bands, self.height, self.width)
        elif self.data_format == "channels_last":
            return (self.height, self.width, self.num_bands)

    def output_layer(self) -> tf.Laye:
        outputDeep = self.deep_neural_network(
            n_filters=16,
            dropout=0.2,
            batchnorm=True,
            data_format="channels_last",
            activation_middle="relu",
            activation_end="softmax",
            kernel_size=3,
            padding="same",
            num_classes=self.num_classes,
        )
        outputDeep = self.Conv(
            self.num_classes,
            3,
            activation="softmax",
            data_format="channels_last",
            padding="same",
        )(outputDeep)
        if self.padding > 0:
            outputDeep = Cropping2D(
                cropping=(
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                )
            )(outputDeep)
        return outputDeep


class UNet3D(UNet):
    def __init__(
        self,
        **kwargs,
    ):
        self.Conv = Conv3D
        self.ConvTranspose = Conv3DTranspose
        self.SpatialDropout = SpatialDropout3D
        self.MaxPooling = MaxPooling3D
        super().__init__(**kwargs)

    @cached_property
    def model_input_shape(self):
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

    def output_layer(self) -> tf.Layer:

        outputDeep = self.deep_neural_network(
            n_filters=16,
            dropout=0.2,
            batchnorm=True,
            data_format="channels_last",
            activation_middle="relu",
            activation_end="softmax",
            kernel_size=3,
            padding="same",
            num_classes=self.num_classes,
        )

        outputs = Conv3D(
            1,
            3,
            activation="relu",
            data_format="channels_first",
            padding="same",
        )(outputDeep)

        outputs2 = self.Conv(
            self.num_classes,
            3,
            activation="softmax",
            data_format="channels_last",
            padding="same",
        )(outputs)

        outputDeep = tf.keras.backend.squeeze(outputs2, 1)
        if self.padding > 0:
            outputDeep = Cropping2D(
                cropping=(
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                )
            )(outputDeep)
        return outputDeep
