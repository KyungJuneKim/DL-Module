import numpy as np
from random import randrange
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from typing import List

from DataSet import DataSet
from Model import Model
from util import plot_model


class RegressionPWM(DataSet):
    def __init__(self, ratio: List[float] = None, period: int = 20, cycle: int = 5):
        self.period = period
        self.cycle = cycle
        if ratio is None:
            ratio = np.arange(0., 1., 0.01).tolist()
        super().__init__(ratio, period * cycle, 1)

    def single_x(self, factor):
        data = []
        scaling = 1000
        for i in range(self.period):
            if i < factor * self.period:
                data.append((800+randrange(200))/scaling)
            else:
                data.append(randrange(200)/scaling)

        return data * self.cycle

    def single_y(self, factor):
        return factor

    def reshape_x(self, x) -> np.ndarray:
        return np.array(x, dtype=np.float32).reshape((-1, self.input_size, 1))

    def reshape_y(self, y) -> np.ndarray:
        return np.array(y, dtype=np.float32)


class RegressionLSTM(Model):
    def __init__(self, data_set: DataSet, epoch: int, learning_rate: float, lstm_size: int):
        super().__init__(data_set, epoch, learning_rate)
        self.lstm_size = lstm_size

    def build_model(self):
        self.add(LSTM(self.lstm_size))
        self.add(Dense(self.data_set.output_size, activation=keras.activations.sigmoid))

        self.compile(loss='mse',
                     optimizer=keras.optimizers.Adam(self.learning_rate),
                     metrics=['mae'])


if __name__ == '__main__':
    data = RegressionPWM(
        period=20,
        cycle=5
    ).make_data(
        num=50,
        ratio=[0.7, 0.2]
    )

    model = RegressionLSTM(
        data_set=data,
        epoch=5,
        learning_rate=0.01,
        lstm_size=200
    )
    model.build_model()

    history = model.fit(
        data.x_train, data.y_train,
        batch_size=1,
        epochs=model.epoch,
        validation_data=(data.x_val, data.y_val)
    )

    loss, mae = model.evaluate(data.x_test, data.y_test)
    print(loss, mae)

    pred = model.predict(
        data.reshape_x(data.single_x(0.75))
    )
    print(pred)

    plot_model(history, validation=True, keys=['loss', 'mae'])
