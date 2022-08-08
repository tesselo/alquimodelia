import keras_unet
import tensorflow as tf
from tensorflow.keras.layers import Cropping2D, Dense, Input
from tensorflow.keras.models import Model, Sequential, load_model

from alquimodelia.alquimodelia_as_class import resnet_arch, unet_arch, unet_arch_2D

# for classes sigmoid sofmax stupid

# for linear linear


# Start easy with imports from open source code
# https://github.com/karolzak/keras-unet/tree/master/keras_unet
# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/maxvfischer/keras-image-segmentation-loss-functions


# Class legacy of ModelMagia. Keeping it to see how to make this as a class

class ModelMagia(tf.keras.Model):
    """docstring for ModelMagia."""

    def __init__(
        self, model_type, timesteps, width, height, padding, num_bands, num_classes
    ):
        super(ModelMagia, self).__init__()
        self.model_type = model_type
        self.timesteps = timesteps
        self.width = width
        self.height = height
        self.padding = padding
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.inside_upsampling = False
        #self.define_model()

    
    def call(self, inputs):
        print('satr')
        i, o  = self.build_3D_UNET(                self.timesteps,
                self.width,
                self.height,
                self.padding,
                self.num_bands,
                self.num_classes)
        #x = self.define_model()
        x =  Model(inputs=i, outputs=o)
        print(x)
        return x
    

    def define_model(self):
        print(self.model_type)

        if self.model_type == "3D_UNET":
            print(self.width)
            print(type(self.num_bands))
            model = self.build_3D_UNET(
                self.timesteps,
                self.width,
                self.height,
                self.padding,
                self.num_bands,
                self.num_classes,
            )
            #model.summary()
            #print(model.to_json())
        if self.model_type == "2D_UNET":
            print("sss")
            model = self.build_2D_UNET(10, 10, 8, 10, 4)
            model.summary()
            print(model.to_json())
        if self.model_type == "ResNet":
            print("resnet")
            X = tf.random.uniform(shape=(1, 128, 128, 10))
            for layer in self.res_net().layers:
                print(layer)
                # X = layer(X)
                print(layer.__class__.__name__, "output shape:\t", X.shape)
            # model = self.res_net(128, 128, 6, 10, 1)
            # model.summary()
            # print(model.to_json())
        return model

    def build_3D_UNET(self, timesteps, width, height, padding, num_bands, num_classes):
        input_shape = (
            timesteps,
            width,
            height,
            num_bands,
        )
        input_img = Input(shape=input_shape, name="input")
        print(input_img.shape)

        # input2_img = unet_arch.inside_upsampling3D(input_img, num_bands, 10)
        input2_img = input_img
        # if padding>0:
        #    input2_img = tf.keras.layers.ZeroPadding3D(padding=padding)(input2_img)
        outputDeep = unet_arch.get_unet_12t(
            input2_img,
            n_filters=16,
            dropout=0.2,
            batchnorm=True,
            data_format="channels_last",
            activation_middle="relu",
            activation_end="softmax",
            kernel_size=3,
            padding="same",
            num_classes=num_classes,
        )
        if padding > 0:
            outputDeep = Cropping2D(cropping=((padding, padding), (padding, padding)))(
                outputDeep
            )
        print("outputDeep", outputDeep.shape)
        return input_shape, outputDeep
        return Model(inputs=input_img, outputs=outputDeep)

    def build_2D_UNET(self, width, height, padding, num_bands, num_classes):
        input_shape = (width + (padding * 2), height + (padding * 2), num_bands)
        input_img = Input(shape=input_shape, name="input")
        print(input_img.shape)
        if self.inside_upsampling:
            input_img = unet_arch.inside_upsampling(input_img, num_bands, 10)
        outputDeep = unet_arch_2D.get_unet_12t(
            input_img,
            n_filters=16,
            dropout=0.2,
            batchnorm=True,
            num_classes=num_classes,
            data_format="channels_last",
            activation_end="softmax",
            activation_middle="relu",
            kernel_size=3,
            padding="same",
        )
        outputDeep = Cropping2D(cropping=((padding, padding), (padding, padding)))(
            outputDeep
        )
        model = Model(inputs=input_img, outputs=outputDeep)
        return model

    # Recall that we define this as a function so we can reuse later and run it
    # within `tf.distribute.MirroredStrategy`'s scope to utilize various
    # computational resources, e.g. GPUs. Also note that even though we have
    # created b1, b2, b3, b4, b5 but we will recreate them inside this function's
    # scope instead
    def res_net():
        return tf.keras.Sequential(
            [
                # The following layers are the same as b1 that we created earlier
                tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
                # The following layers are the same as b2, b3, b4, and b5 that we
                # created earlier
                resnet_arch.ResnetBlock(64, 2, first_block=True),
                resnet_arch.ResnetBlock(128, 2),
                resnet_arch.ResnetBlock(256, 2),
                resnet_arch.ResnetBlock(512, 2),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Dense(units=10),
            ]
        )
