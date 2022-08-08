from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    concatenate,
)
from tensorflow.keras.models import Model


# Pixel model
def Pixel_model(
    timesteps,
    bands,
    classes,
    activation_final=None,
    return_last_layer=False,
    classifyer=True,
):
    # input layer
    visible = Input(shape=(timesteps, bands))
    normed = BatchNormalization()(visible)

    # first feature extractor
    conv1 = Conv1D(filters=64, kernel_size=3, activation="relu")(normed)
    normed1 = BatchNormalization()(conv1)
    dropped1 = Dropout(0.25)(normed1)
    convd1 = Conv1D(filters=64, kernel_size=3, activation="relu")(dropped1)
    normed11 = BatchNormalization()(convd1)
    pool1 = MaxPooling1D(pool_size=2)(normed11)
    flat1 = Flatten()(pool1)

    min_value_inshape = min([f for f in normed.shape if f is not None])
    min_value_div = [f for f in normed.shape if f is not None]
    min_value_div = [f for f in min_value_div if not (f / 2) % 2]
    values_consider = []

    for i in range(min_value_inshape):
        step = min_value_inshape - i + 1
        if step >= i:
            values_consider.append(i)

    kernel_size_second = max(values_consider)

    # second feature extractor
    conv2 = Conv1D(filters=64, kernel_size=kernel_size_second, activation="relu")(
        normed
    )
    normed2 = BatchNormalization()(conv2)

    dropped2 = Dropout(0.25)(normed2)
    convd2 = Conv1D(filters=64, kernel_size=kernel_size_second, activation="relu")(
        dropped2
    )
    normed22 = BatchNormalization()(convd2)
    pool_zi = min([f for f in normed22.shape if f is not None])
    pool_zi = min([pool_zi, 2])
    pool2 = MaxPooling1D(pool_size=pool_zi)(normed22)
    flat2 = Flatten()(pool2)
    # merge feature extractors
    merge = concatenate([flat1, flat2])
    dropped = Dropout(0.5)(merge)
    if return_last_layer:
        return visible, dropped
    # interpretation layer
    hidden1 = Dense(100, activation="relu")(dropped)
    normed3 = BatchNormalization()(hidden1)

    if activation_final is None:
        if classifyer is True:
            activation_final = "softmax"
        else:
            activation_final = "linear"

    # prediction output
    output = Dense(classes, activation=activation_final)(normed3)
    pixel_model = Model(inputs=visible, outputs=output)
    return pixel_model
