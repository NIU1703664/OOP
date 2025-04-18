from typing import Self, TypeVar, Generic
import math
import numpy as np
import numpy.typing as npt
import logging
Label = TypeVar("Label", bound=np.generic, covariant=True)

# Configurate the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Dataset(Generic[Label]):
    def __init__ (self, X: npt.NDArray[np.float64], y: npt.NDArray[Label]):
        self.X: npt.NDArray[np.float64] = X;
        assert self.X.ndim == 2;
        self.y: npt.NDArray[Label] = y;
        assert self.X.ndim == 1;
        self.num_samples: int
        self.num_features: int
        self.num_samples, self.num_features=self.X.shape

    def random_sampling(self,ratio_samples: float) -> Self:
        n:int = math.floor(self.num_features*ratio_samples)
        indexes: list[int]= list(np.random.choice(range(0,n),int(n*ratio_samples), replace =True));
        return type(self)( np.array([self.X[i] for i in indexes]), np.array([self.y[i] for i in indexes]))

    def most_frequent_label(self) -> Label:
        values, counts = np.unique(self.y, return_counts=True)
        ind = np.argmax(counts)
        ret: Label = values[ind]
        return ret

    def split(self, feature_index: np.int64, value: np.float64) -> tuple[Self, Self]:
        left_X: list[list[np.float64]] = []
        left_y: npt.NDArray[Label] = np.array([])
        right_X: list[list[np.float64]] = []
        right_y: npt.NDArray[Label] = np.array([])
        for i in range(self.num_samples):
            if self.X[feature_index] < value:
                left_X[i] = self.X[i]
                left_y[i] = self.y[i]
            else:
                right_X[i] = self.X[i]
                right_y[i] = self.y[i]
        return (type(self)(np.array(left_X), left_y), type(self)(np.array(right_X), right_y))
