import numpy as np
from abc import *
from typing import List, Tuple

from util import split_data_set


class DataSet(metaclass=ABCMeta):
    def __init__(self, factors: List, input_size, output_size):
        self.factors = factors
        self.input_size = input_size
        self.output_size = output_size

        self.data_sets: List[Tuple[np.ndarray, np.ndarray]] = []

    @property
    def x_train(self) -> np.ndarray:
        return self.data_sets[0][0] if len(self.data_sets) > 0 else None

    @property
    def y_train(self) -> np.ndarray:
        return self.data_sets[0][1] if len(self.data_sets) > 0 else None

    @property
    def x_val(self) -> np.ndarray:
        return self.data_sets[1][0] if len(self.data_sets) > 1 else None

    @property
    def y_val(self) -> np.ndarray:
        return self.data_sets[1][1] if len(self.data_sets) > 1 else None

    @property
    def x_test(self) -> np.ndarray:
        return self.data_sets[2][0] if len(self.data_sets) > 2 else None

    @property
    def y_test(self) -> np.ndarray:
        return self.data_sets[2][1] if len(self.data_sets) > 2 else None

    def generate(self, num: int, ratio: List[float]):
        if not ratio:
            raise AttributeError('Nothing in ratio')

        self.data_sets.clear()
        data_sets = [[[], []] for _ in range(len(ratio)+1)]

        for factor in self.factors:
            x = []
            y = []
            for i in range(num):
                x.append(self.single_x(factor))
                y.append(self.single_y(factor))

            split_sets = split_data_set(x, y, ratio)
            for data_set, split_set in zip(data_sets, split_sets):
                data_set[0] += split_set[0]
                data_set[1] += split_set[1]

        for data_set in data_sets:
            self.data_sets.append(
                (self.reshape_x(data_set[0]), self.reshape_y(data_set[1]))
            )

        return self

    @abstractmethod
    def single_x(self, factor):
        pass

    @abstractmethod
    def single_y(self, factor):
        pass

    @abstractmethod
    def reshape_x(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def reshape_y(self, y) -> np.ndarray:
        pass
