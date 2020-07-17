from abc import *
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from typing import List, Optional
from warnings import warn

import Data


class Model(metaclass=ABCMeta):
    def __init__(self, model: keras.Sequential, data_set: Data, epoch: int, learning_rate: float):
        self.model: Optional[keras.Sequential] = model
        self.data_set: Data = data_set

        self.epoch: int = epoch
        self.learning_rate: float = learning_rate

        self.data_path: str = ''
        self.save_path: str = 'model'

    def save_model(self):
        self.model.save(self.save_path)


class CategoricalLSTM(Model):
    def __init__(self, data_set: Data, epoch: int, learning_rate: float, lstm_size: int):
        self.lstm_size = lstm_size
        m = keras.Sequential()
        m.add(LSTM(lstm_size))
        m.add(Dense(data_set.output_size, activation=keras.activations.softmax))

        m.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.RMSprop(super().learning_rate),
                  metrics=['mse', 'accuracy'])

        super().__init__(m, data_set, epoch, learning_rate)
