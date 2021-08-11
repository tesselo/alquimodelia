import tensorflow as tf
from alquimodelia import unet_arch, unet_arch_2D, resnet_arch
from tensorflow.keras.layers import Input, Dense, Cropping2D
from tensorflow.keras.models import Model, Sequential, load_model



class ModelMagia(tf.keras.Model):
    """docstring for ModelMagia."""

    def __init__(self, model_type):
        super(ModelMagia, self).__init__()
        self.model_type = model_type
        self.call()

    def call(self):
        if self.model_type == "3D_UNET":
            print('sss')
            model = self.build_3D_UNET(8, 128, 128, 6, 10, 1)
            #model.summary()
            print(model.to_json())
        if self.model_type == "2D_UNET":
            print('sss')
            model = self.build_2D_UNET(128, 128, 6, 10, 1)
            model.summary()
            print(model.to_json())
        if self.model_type == "ResNet":
            print('resnet')
            X = tf.random.uniform(shape=(1, 128, 128, 10))
            for layer in self.res_net().layers:
                print(layer)
                #X = layer(X)
                print(layer.__class__.__name__, 'output shape:\t', X.shape)
            #model = self.res_net(128, 128, 6, 10, 1)
            #model.summary()
            #print(model.to_json())

        return model


    def build_3D_UNET(self, timesteps, width, height, padding, num_bands, num_classes):
        input_shape = (timesteps, width + (padding*2),  height + (padding*2), num_bands)
        input_img = Input(shape=input_shape, name="input")
        outputDeep = unet_arch.get_unet_12t(input_img, n_filters=16, dropout=0.05, batchnorm=True, data_format='channels_last',
                                       activation_end='sigmoid',
                                       kernel_size=3, padding='same', num_classes=num_classes, local=False)
        outputDeep = Cropping2D(cropping=((padding, padding), (padding, padding)))(outputDeep)
        model = Model(inputs=input_img, outputs=outputDeep)
        return model

    def build_2D_UNET(self, width, height, padding, num_bands, num_classes):
        input_shape = (width + (padding*2),  height + (padding*2), num_bands)
        input_img = Input(shape=input_shape, name="input")
        outputDeep = unet_arch_2D.get_unet_10b_2D(input_img, n_filters=8, dropout=0.15, batchnorm=True,
                                       num_classes=num_classes)
        outputDeep = Cropping2D(cropping=((padding, padding), (padding, padding)))(outputDeep)
        model = Model(inputs=input_img, outputs=outputDeep)
        return model


    # Recall that we define this as a function so we can reuse later and run it
    # within `tf.distribute.MirroredStrategy`'s scope to utilize various
    # computational resources, e.g. GPUs. Also note that even though we have
    # created b1, b2, b3, b4, b5 but we will recreate them inside this function's
    # scope instead
    def res_net():
        return tf.keras.Sequential([
            # The following layers are the same as b1 that we created earlier
            tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            # The following layers are the same as b2, b3, b4, and b5 that we
            # created earlier
            resnet_arch.ResnetBlock(64, 2, first_block=True),
            resnet_arch.ResnetBlock(128, 2),
            resnet_arch.ResnetBlock(256, 2),
            resnet_arch.ResnetBlock(512, 2),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=10)])