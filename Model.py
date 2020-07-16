from abc import *
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense

import Data


class Model(metaclass=ABCMeta):
    def __init__(self, epoch: int, learning_rate: float, data_set: Data):
        self.epoch = epoch
        self.learning_rate = learning_rate

        self.data_set = data_set

        self.data_path = ''
        self.save_path = ''

    @abstractmethod
    def build_model(self) -> keras.Sequential:
        pass

    def save_model(self):
        pass


class CategoricalLSTM(Model):
    def __init__(self, epoch: int, learning_rate: float, data_set: Data, lstm_size: int):
        super().__init__(epoch, learning_rate, data_set)
        self.lstm_size = lstm_size

    def build_model(self) -> keras.Sequential:
        m = keras.Sequential()
        m.add(LSTM(self.lstm_size))
        m.add(Dense(super().data_set.output_size, activation=keras.activations.softmax))

        m.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.RMSprop(super().learning_rate),
                  metrics=['mse', 'accuracy'])

        return m
