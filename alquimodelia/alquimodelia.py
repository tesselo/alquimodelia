import keras_unet

from alquimodelia.pixel import Pixel_model

# for classes sigmoid sofmax stupid


# Start easy with imports from open source code
# https://github.com/karolzak/keras-unet/tree/master/keras_unet
# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/maxvfischer/keras-image-segmentation-loss-functions


def ModelMagia(
    model_type,
    timesteps,
    width,
    height,
    padding,
    num_bands,
    num_classes,
    activation_final=None,
):
    if model_type == "pixel":
        model = Pixel_model(
            timesteps,
            num_bands,
            num_classes,
            activation_final=None,
            return_last_layer=False,
            classifyer=True,
        )
    elif model_type == "3D_UNET":
        model = "ss"
    elif model_type == "2D_UNET":
        model = "ss"
    elif "keras_unet" in model_type:
        # Use keras_unet.name_of_model
        model_name = model_type.replace("keras_unet.", "")
        model = getattr(keras_unet.models, model_name)
        model = model(
            input_shape=(width, height, num_bands),
            use_batch_norm=True,
            num_classes=num_classes,
            filters=64,
            dropout=0.2,
            output_activation=activation_final,
        )
    return model
