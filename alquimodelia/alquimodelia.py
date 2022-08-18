import tensorflow as tf
from tensorflow.keras.layers import Input

# for classes sigmoid sofmax stupid


# Start easy with imports from open source code
# https://github.com/karolzak/keras-unet/tree/master/keras_unet
# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/maxvfischer/keras-image-segmentation-loss-functions


class ModelMagia(tf.keras.Model):
    def define_input_shape(self):
        raise NotImplementedError

    def get_input_layer(self):
        self.define_input_shape()
        self.input_layer = Input(self.model_input_shape)

    def get_output_layer(self):
        raise NotImplementedError

    def get_last_layer_activation(self):
        raise NotImplementedError

    def __init__(
        self,
        timesteps=None,
        width=None,
        height=None,
        padding=None,
        num_bands=None,
        num_classes=None,
        activation_final=None,
    ):

        self.num_classes = num_classes
        self.timesteps = timesteps
        self.width = width
        self.height = height
        self.padding = padding
        self.num_bands = num_bands
        self.activation_final = activation_final
        self.get_input_layer()

        super().__init__(
            inputs=self.input_layer,
            outputs=self.get_output_layer(),
        )
