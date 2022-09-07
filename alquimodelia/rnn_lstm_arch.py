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
        dropout: float = 0.3,
        recurrent_dropout: float = None,
        activation_middle: str = "relu",
        pool_size: int = 2,
        **kwargs,
    ):
        self.lstm_units = lstm_units
        self.dropout = dropout
        if recurrent_dropout is None:
            self.recurrent_dropout = self.dropout
        else:
            self.recurrent_dropout = self.recurrent_dropout
        self.activation_middle = activation_middle
        self.pool_size = pool_size
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
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
        )(x)
        x = MaxPooling1D(pool_size=self.pool_size)(x)
        x = BatchNormalization()(x)
        x = Bidirectional(
            LSTM(
                self.lstm_units[1],
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)
        x = Dense(120, activation=self.activation_middle)(x)
        x = BatchNormalization()(x)
        x = Dense(self.num_classes, activation=self.activation_final)(x)
        return x
