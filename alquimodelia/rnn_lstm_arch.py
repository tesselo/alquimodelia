from functools import cached_property
from typing import Tuple

from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    MaxPooling1D,
)

from alquimodelia.alquimodelia import ModelMagia


class RnnLSTM(ModelMagia):
    """Base class for LSTM Pixel models"""

    def __init__(
        self,
        lstm_units: Tuple[int, int] = (120, 80),
        **kwargs,
    ):
        self.lstm_units = lstm_units
        super().__init__(**kwargs)

    @cached_property
    def model_input_shape(self):
        if self.data_format == "channels_first":
            return (
                self.num_bands,
                self.timesteps,
            )
        elif self.data_format == "channels_last":
            return (self.timesteps, self.num_bands)

    def output_layer(self):
        x = BatchNormalization()(self.input_layer)
        x = LSTM(
            self.lstm_units[0],
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
        )(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(self.lstm_units[1], dropout=0.2, recurrent_dropout=0.2))(
            x
        )
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(120, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(self.num_classes, activation=self.activation_final)(x)
        return x
