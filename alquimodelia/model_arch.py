from tensorflow.keras.layers import (
    GRU,
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Non-sequential model with shared Input Layer
# https://machinelearningmastery.com/keras-functional-api-deep-learning/

# Input layer
visible = Input(shape=(44, 10))
normed = BatchNormalization()(visible)
# First 1D feature extractor.
conv1 = Conv1D(filters=64, kernel_size=3, activation="relu")(normed)
normed1 = BatchNormalization()(conv1)
dropped1 = Dropout(0.25)(normed1)
convd1 = Conv1D(filters=64, kernel_size=3, activation="relu")(dropped1)
normed11 = BatchNormalization()(convd1)
pool1 = MaxPooling1D(pool_size=2)(normed11)
flat1 = Flatten()(pool1)
# Second 1D feature extractor.
conv2 = Conv1D(filters=64, kernel_size=6, activation="relu")(normed)
normed2 = BatchNormalization()(conv2)
dropped2 = Dropout(0.25)(normed2)
convd2 = Conv1D(filters=64, kernel_size=6, activation="relu")(dropped2)
normed22 = BatchNormalization()(convd2)
pool2 = MaxPooling1D(pool_size=2)(normed22)
flat2 = Flatten()(pool2)
# GRU Model.
gru1 = GRU(100, return_sequences=False, recurrent_dropout=0.2)(normed)
# Merge 1D models.
merge = concatenate([flat1, flat2, gru1])
dropped = Dropout(0.5)(merge)
# Interpretation layer.
hidden1 = Dense(100, activation="relu")(dropped)
# Normalize mixing.
normed3 = BatchNormalization()(hidden1)
# Prediction output.
output = Dense(7, activation="softmax")(normed3)
model = Model(inputs=visible, outputs=output)
# Summarize layers
model.summary()
# plot_model(model, to_file='/home/tam/Desktop/shared_input_layer.png')
model.to_json()
