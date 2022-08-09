import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    Cropping2D,
    Input,
    MaxPooling2D,
    MaxPooling3D,
    SpatialDropout2D,
    SpatialDropout3D,
    concatenate,
)
from tensorflow.keras.models import Model

# momentum must be big so the different batches dont do much damage
# last conv must have a stride and kernel of 1, with a density of num classes

# For classification sigmoid or softmax in last layer
# RELU in the middle
# linear in last for linear

# Number of steps must be a log2(size) || actually max number should be when the small filter has 8 px so the 3 kernel still makes sense

# Try dinamic filters
# They change at each step
# They stay the same all way
# different for contract and for expandig
#

# Try attention gate
#

# dafuq is this UpSampling2D?


def count_number_divisions(size, count, by=2, limit=8):
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


class UNET(tf.keras.Model):
    def __init__(
        self,
        timesteps,
        width,
        height,
        padding,
        num_bands,
        num_classes,
        dimension="3D",
        number_of_conv_layers=None,
    ):
        self.num_classes = num_classes
        self.dimension = dimension
        self.timesteps = timesteps
        self.width = width
        self.height = height
        self.padding = padding
        self.num_bands = num_bands
        self.number_of_conv_layers = number_of_conv_layers

        # Set the dimensinal operations.
        if dimension == "3D":
            self.Conv = Conv3D
            self.ConvTranspose = Conv3DTranspose
            self.SpatialDropout = SpatialDropout3D
            self.MaxPooling = MaxPooling3D
            self.model_input_shape = (timesteps, height, width, num_bands)
        elif dimension == "2D":
            self.Conv = Conv2D
            self.ConvTranspose = Conv2DTranspose
            self.SpatialDropout = SpatialDropout2D
            self.MaxPooling = MaxPooling2D
            self.model_input_shape = (height, width, num_bands)
            self.timesteps = 1
        else:
            print("Dimension not available")

    def convblock(self, x):
        return self.Conv(x)

    def convolution_block(
        self,
        input_tensor,
        n_filters,
        kernel_size=3,
        batchnorm=True,
        data_format="channels_first",
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
    ):
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
            activation=activation,
            data_format=data_format,
        )(x)
        if batchnorm:
            x = BatchNormalization()(x)
        return x

    def contracting_block(
        self,
        input_img,
        n_filters,
        batchnorm,
        dropout=0.5,
        kernel_size=3,
        strides=2,
        data_format="channels_last",
        padding="same",
        activation="relu",
    ):
        c1 = self.convolution_block(
            input_img,
            n_filters=n_filters * 1,
            kernel_size=kernel_size,
            batchnorm=batchnorm,
            data_format=data_format,
            activation=activation,
            padding=padding,
        )
        p1 = self.MaxPooling(strides, padding=padding)(c1)
        p1 = self.SpatialDropout(dropout * 0.5, data_format=data_format)(p1)
        return p1, c1

    def expansive_block(
        self,
        ci,
        cii,
        n_filters,
        batchnorm,
        dropout=0.5,
        kernel_size=3,
        strides=2,
        data_format="channels_first",
        activation="relu",
        padding="same",
    ):
        u = self.ConvTranspose(
            n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
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
            padding=padding,
        )
        return c

    def get_unet_12t(
        self,
        input_img,
        n_filters=16,
        dropout=0.2,
        batchnorm=True,
        data_format="channels_last",
        activation_end="sigmoid",
        activation_middle="relu",
        kernel_size=3,
        padding="same",
        num_classes=1,
    ):
        # contracting path
        p1, c1 = self.contracting_block(
            input_img,
            n_filters,
            batchnorm,
            dropout=dropout,
            kernel_size=kernel_size,
            data_format=data_format,
            activation=activation_middle,
            padding=padding,
        )
        print(f"p : {p1.shape}")
        print(f"c : {c1.shape}")

        p2, c2 = self.contracting_block(
            p1,
            n_filters * 2,
            batchnorm,
            dropout=dropout,
            kernel_size=kernel_size,
            data_format=data_format,
            activation=activation_middle,
            padding=padding,
        )
        print(f"p : {p2.shape}")
        print(f"c : {c2.shape}")

        # middle
        p3, c3 = self.contracting_block(
            p2,
            n_filters * 4,
            batchnorm,
            dropout=dropout,
            kernel_size=kernel_size,
            data_format=data_format,
            activation=activation_middle,
            padding=padding,
        )
        print(f"p : {p3.shape}")
        print(f"c : {c3.shape}")

        # expansive path
        c4 = self.expansive_block(
            c3,
            c2,
            n_filters * 2,
            batchnorm,
            dropout=dropout,
            data_format=data_format,
            activation=activation_middle,
            kernel_size=kernel_size,
        )
        print(f"c : {c4.shape}")

        c5 = self.expansive_block(
            c4,
            c1,
            n_filters * 1,
            batchnorm,
            dropout=dropout,
            data_format=data_format,
            activation=activation_middle,
            kernel_size=kernel_size,
        )
        print(f"c455555555 : {c5.shape}")
        if self.dimension == "3D":
            outputs = Conv3D(
                1,
                3,
                activation=activation_middle,
                data_format="channels_first",
                padding=padding,
            )(c5)
        else:
            outputs = c5

        outputs2 = self.Conv(
            num_classes,
            3,
            activation=activation_end,
            data_format="channels_last",
            padding=padding,
        )(outputs)
        if self.dimension == "3D":
            outputs2 = tf.keras.backend.squeeze(outputs2, 1)

        return outputs2

    def unet(self):

        # Set the number of steps.
        # This will build the max number of steps but sometimes the max is not the best.
        number_of_layers = []
        for size in self.model_input_shape:
            number_of_layers.append(count_number_divisions(size, 0))
        print(number_of_layers)
        if self.number_of_conv_layers is None:
            self.number_of_conv_layers = max(number_of_layers)

        # Create the input layer.
        inputs = Input(self.model_input_shape)

        outputDeep = self.get_unet_12t(
            inputs,
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
        if self.padding > 0:
            outputDeep = Cropping2D(
                cropping=((self.padding, self.padding), (self.padding, self.padding))
            )(outputDeep)

        return Model(inputs=inputs, outputs=outputDeep)
