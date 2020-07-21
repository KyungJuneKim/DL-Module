from abc import *
from tensorflow.keras import activations, losses, metrics, optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM


class Model(Sequential, metaclass=ABCMeta):
    def __init__(
            self,
            input_size: int, output_size: int,
            epoch: int, learning_rate: float
    ):
        super().__init__()
        self.input_size: int = input_size
        self.output_size: int = output_size

        self.epoch: int = epoch
        self.learning_rate: float = learning_rate

        self.data_path: str = ''
        self.save_path: str = 'model'

    @abstractmethod
    def build_model(self):
        pass


class CategoricalLSTM(Model):
    def __init__(
            self,
            input_size: int, output_size: int,
            epoch: int, learning_rate: float,
            lstm_size: int
    ):
        super().__init__(
            input_size, output_size,
            epoch, learning_rate
        )
        self.lstm_size = lstm_size

    def build_model(self):
        self.add(LSTM(self.lstm_size))
        self.add(Dense(self.output_size, activation=activations.softmax))

        self.compile(
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizers.RMSprop(self.learning_rate),
            metrics=[
                metrics.categorical_accuracy,
                metrics.mean_squared_error
            ]
        )


class ModelInit(Sequential, metaclass=ABCMeta):
    def __init__(
            self,
            input_size: int, output_size: int,
            epoch: int, learning_rate: float
    ):
        super().__init__()
        self.input_size: int = input_size
        self.output_size: int = output_size

        self.epoch: int = epoch
        self.learning_rate: float = learning_rate

        self.data_path: str = ''
        self.save_path: str = 'model'


class CategoricalLSTMInit(ModelInit):
    def __init__(
            self,
            input_size: int, output_size: int,
            epoch: int, learning_rate: float,
            lstm_size: int
    ):
        super().__init__(
            input_size, output_size,
            epoch, learning_rate
        )
        self.lstm_size = lstm_size
        self.add(LSTM(lstm_size))
        self.add(Dense(output_size, activation=activations.softmax))

        self.compile(
            loss=losses.CategoricalCrossentropy(),
            optimizer=optimizers.RMSprop(learning_rate),
            metrics=[
                metrics.categorical_accuracy,
                metrics.mean_squared_error
            ]
        )
