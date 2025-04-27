from typing import Self, TypeVar, Generic
import math
import numpy as np
import pandas as pd
import sklearn.datasets
import numpy.typing as npt
import logging

# Configurate the logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


class Dataset:
    def __init__(self, X: npt.NDArray[np.float64], y: npt.NDArray[str]):
        self.X: npt.NDArray[np.float64] = X
        assert self.X.ndim == 2
        self.y: npt.NDArray[str] = y
        assert self.y.ndim == 1
        self.num_samples: int
        self.num_features: int
        self.num_samples, self.num_features = self.X.shape

    def random_sampling(self, ratio_samples: float) -> Self:
        n: int = math.floor(self.num_features * ratio_samples)
        indexes: list[int] = list(
            np.random.choice(range(0, n), int(n * ratio_samples), replace=True)
        )
        return type(self)(
            np.array([self.X[i] for i in indexes]),
            np.array([self.y[i] for i in indexes]),
        )

    def most_frequent_label(self) -> str:
        values, counts = np.unique(self.y, return_counts=True)
        ind = np.argmax(counts)
        ret: str = values[ind]
        return ret

    def split(
        self, feature_index: np.int64, value: np.float64
    ) -> tuple[Self, Self]:
        left_X: list[list[np.float64]] = []
        left_y: npt.NDArray[str] = np.array([])
        right_X: list[list[np.float64]] = []
        right_y: npt.NDArray[str] = np.array([])
        for i in range(self.num_samples):
            if self.X[feature_index] < value:
                left_X[i] = self.X[i]
                left_y[i] = self.y[i]
            else:
                right_X[i] = self.X[i]
                right_y[i] = self.y[i]
        return (
            type(self)(np.array(left_X), left_y),
            type(self)(np.array(right_X), right_y),
        )

    @classmethod
    def load_sonar(cls) -> Self:
        df = pd.read_csv('sonar.all-data', header=None)
        if df.empty:
            return False

        X: npt.NDArray[np.float64] = df[df.columns[:-1]].to_numpy(
            dtype=np.float64
        )
        assert self.X.ndim == 2

        y: npt.NDArray[str] = df[df.columns[-1]].to_numpy(dtype=str)
        y = (y == 'M').astype(int)   # M = mine, R = rock

        return cls(X, y)

    @classmethod
    def load_iris(cls) -> Self:
        X: npt.NDArray[np.float64]
        y: npt.NDArray[str]
        X, y = sklearn.datasets.load_iris(return_X_y=True)

        return cls(X, y)
