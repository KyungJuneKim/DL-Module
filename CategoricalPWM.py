import numpy as np
from random import randrange
from tensorflow.keras import activations, losses, metrics, optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Any, List

from DataSet import DataSet
from util import plot_model


class CategoricalPWM(DataSet):
    def __init__(self, ratio: List[float] = None, period: int = 20, cycle: int = 5):
        self.period = period
        self.cycle = cycle
        if not ratio:
            ratio = [0, 0.25, 0.5, 0.75, 1]
        super().__init__(ratio, period * cycle, len(ratio))

    def _get_raw_data(self) -> Any:
        return None

    def single_x(self, factor):
        data = []
        scaling = 1000
        for i in range(self.cycle):
            for j in range(self.period):
                if j < factor * self.period:
                    data.append((800+randrange(200))/scaling)
                else:
                    data.append(randrange(200)/scaling)

        return data

    def single_y(self, factor):
        return self.factors.index(factor)

    def reshape_x(self, x) -> np.ndarray:
        return np.array(x, dtype=np.float32).reshape((-1, self.input_size, 1))

    def reshape_y(self, y) -> np.ndarray:
        return np.eye(len(self.factors), dtype=np.float32)[y]


class CategoricalLSTM(Sequential):
    def __init__(
            self,
            input_size: int, output_size: int,
            epoch: int, batch_size,
            learning_rate: float, lstm_size: int
    ):
        super(CategoricalLSTM, self).__init__()
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.lstm_size: int = lstm_size

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

        return self


if __name__ == '__main__':
    data = CategoricalPWM(
        ratio=[0., .25, .5, .75, 1.],
        period=20,
        cycle=5
    ).generate(
        num=100,
        ratio=[0.7, 0.2]
    )

    model = CategoricalLSTM(
        input_size=data.input_size,
        output_size=data.output_size,
        epoch=5,
        batch_size=1,
        learning_rate=0.01,
        lstm_size=50
    )
    model.build_model()

    history = model.fit(
        data.x_train, data.y_train,
        batch_size=model.batch_size,
        epochs=model.epoch,
        validation_data=(data.x_val, data.y_val)
    )

    plot_model(history)

    loss, accuracy, mse = model.evaluate(data.x_test, data.y_test)
    print(loss, accuracy, mse)

    pred = model.predict(
        data.reshape_x(data.single_x(0.75))
    )
    print(pred)
