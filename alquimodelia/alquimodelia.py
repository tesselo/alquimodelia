import tensorflow as tf
from tensorflow.keras.layers import Input

# for classes sigmoid sofmax stupid


# Start easy with imports from open source code
# https://github.com/karolzak/keras-unet/tree/master/keras_unet
# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/maxvfischer/keras-image-segmentation-loss-functions


class ModelMagia(tf.keras.Model):
    def model_input_shape(self):
        raise NotImplementedError

    def define_input_layer(self):
        self.input_layer = Input(self.model_input_shape)

    def output_layer(self):
        raise NotImplementedError

    def get_last_layer_activation(self):
        raise NotImplementedError

    def __init__(
        self,
        timesteps: int = 0,
        width: int = 0,
        height: int = 0,
        num_bands: int = 0,
        num_classes: int = 0,
        activation_final: str = "sigmoid",
        data_format: str = "channels_last",
    ):

        self.num_classes = num_classes
        self.timesteps = timesteps
        self.width = width
        self.height = height
        self.num_bands = num_bands
        self.activation_final = activation_final
        self.data_format = data_format

        super().__init__(
            inputs=self.input_layer,
            outputs=self.output_layer(),
        )
