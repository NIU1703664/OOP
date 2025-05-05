from typing import Self, TypeVar, Generic
import math
import numpy as np
import pandas as pd
import sklearn.datasets
import pickle
import numpy.typing as npt
import logging

# Configurate the logging
logging.basicConfig(
    level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s'
)


class Dataset:
    def __init__(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int64],
    ):
        self.X: npt.NDArray[np.float64] = X
        assert self.X.ndim == 2
        self.y: npt.NDArray[np.int64] = y
        assert self.y.ndim == 1
        # assert all(type(k) == np.int64 for k in y)
        self.num_samples: int
        self.num_features: int
        self.num_samples, self.num_features = self.X.shape

    def random_sampling(self, ratio_samples: float) -> Self:
        n: int = math.floor(self.num_samples * ratio_samples)
        indexes: list[int] = list(
            np.random.choice(range(0, n), int(n * ratio_samples), replace=True)
        )
        return type(self)(
            np.array([self.X[i] for i in indexes]),
            np.array([self.y[i] for i in indexes]),
        )

    def most_frequent_label(self) -> np.int64:
        values, counts = np.unique(self.y, return_counts=True)
        ind = np.argmax(counts)
        ret: np.int64 = values[ind]
        return ret

    def split(
        self, feature_index: np.int64, value: np.float64
    ) -> tuple[Self, Self]:
        left_length = 0
        for i in range(self.num_samples):
            if self.X[i, feature_index] < value:
                left_length += 1
        left_X: npt.NDArray[np.float64] = np.zeros(
            (left_length, self.num_features)
        )
        left_y: npt.NDArray[np.int64] = np.zeros(left_length, dtype=np.int64)
        right_length = self.num_samples - left_length
        right_X: npt.NDArray[np.float64] = np.zeros(
            (right_length, self.num_features)
        )
        right_y: npt.NDArray[np.int64] = np.zeros(right_length, dtype=np.int64)

        left_index, right_index = 0, 0
        for i in range(self.num_samples):
            if self.X[i, feature_index] < value:
                left_X[left_index], left_y[left_index] = (
                    self.X[i, :],
                    self.y[i],
                )
                left_index += 1

            else:
                right_X[right_index], right_y[right_index] = (
                    self.X[i, :],
                    self.y[i],
                )
                right_index += 1
        assert all(type(k) == np.int64 for k in left_y)
        assert all(type(k) == np.int64 for k in right_y)
        return (
            type(self)(np.array(left_X), left_y),
            type(self)(np.array(right_X), right_y),
        )

    @classmethod
    def load_sonar(cls) -> Self:
        df = pd.read_csv('./datasets/Sonar/sonar.all-data.csv', header=None)

        X: npt.NDArray[np.float64] = df[df.columns[:-1]].to_numpy(
            dtype=np.float64
        )
        assert X.ndim == 2

        y: npt.NDArray[np.int64] = df[df.columns[-1]].to_numpy(dtype=str)

        y = np.array(
            list(map(lambda x: np.int64(x == 'M'), y))
        )   # M = mine, R = rock

        return cls(X, y)

    @classmethod
    def load_iris(cls) -> Self:
        X: npt.NDArray[np.float64]
        y: npt.NDArray[np.int64]
        X, y = sklearn.datasets.load_iris(return_X_y=True)

        # labels: npt.NDArray[np.int64] = np.unique(y)

        return cls(X, y)

    @classmethod
    def load_MNIST(cls) -> Self:
        with open('./datasets/MNIST/mnist.pkl', 'rb') as f:
            mnist = pickle.load(f, encoding='byte')

        images = np.append(
            np.array(mnist['training_images']),
            np.array(mnist['test_images']),
            axis=0,
        )
        labels = np.append(
            np.array(mnist['training_labels']), np.array(mnist['test_labels'])
        )

        return cls(images, labels)

    @classmethod
    def load_temperatures(cls) -> Self:
        df = pd.read_csv(
            'https://raw.githubusercontent.com/jbrownlee/'
            'Datasets/master/daily-min-temperatures.csv'
        )
        day: np.int64 = pd.DatetimeIndex(df.Date).day.to_numpy()   # 1...31
        month: np.int64 = pd.DatetimeIndex(df.Date).month.to_numpy()   # 1...12
        year: np.int64 = pd.DatetimeIndex(
            df.Date
        ).year.to_numpy()   # 1981...1999
        X: npt.NDArray[np.int64] = np.vstack(
            [day, month, year]
        ).T   # np array of 3 columns
        y: npt.NDArray[np.float64] = df.Temp.to_numpy()
        return cls(X, y)
