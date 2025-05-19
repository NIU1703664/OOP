from abc import abstractmethod
import numpy as np
from typing import override
from numpy.typing import NDArray
from .dataset import Dataset
from .measure import Impurity


class Split:
    def __init__(self, criterion: Impurity) -> None:
        self.criterion: Impurity = criterion

    @abstractmethod
    def _best_split(
        self, idx_features: NDArray[np.int64], dataset: Dataset
    ) -> tuple[np.int64, np.float64, np.float64, list[Dataset]]:
        pass

    def _CART_cost(
        self, left_dataset: Dataset, right_dataset: Dataset
    ) -> np.float64:
        # the J(k,v) equation in the slides, using Gini
        left_len = np.float64(left_dataset.num_samples)
        right_len = np.float64(right_dataset.num_samples)
        critereon: Impurity = self.criterion

        return left_len / (left_len + right_len) * critereon.purity(
            left_dataset
        ) + right_len / (left_len + right_len) * critereon.purity(
            right_dataset
        )


class RandomSplit(Split):
    @override
    def _best_split(
        self,
        idx_features: NDArray[np.int64],
        dataset: Dataset,
    ) -> tuple[
        np.int64, np.float64, np.float64, list[Dataset]
    ]:   # find the best pair (feature, threshold) by exploring all possible pairs
        int_max = np.int64(np.iinfo(np.int64).max)
        best_feature_index, best_threshold, minimum_cost = (
            int_max,
            np.float64(np.inf),
            np.float64(np.inf),
        )
        best_split: list[Dataset] | None = None
        idx: np.int64
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            val: np.float64
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset)   # J(k,v)
                if cost < minimum_cost:
                    (
                        best_feature_index,
                        best_threshold,
                        minimum_cost,
                        best_split,
                    ) = (idx, val, cost, [left_dataset, right_dataset])

        if best_split == None or best_feature_index == np.inf:
            raise Exception('Dataset is empty')

        return best_feature_index, best_threshold, minimum_cost, best_split


class ExtraSplit(Split):
    @override
    def _best_split(
        self, idx_features: NDArray[np.int64], dataset: Dataset
    ) -> tuple[
        np.int64, np.float64, np.float64, list[Dataset]
    ]:   # find the best pair (feature, threshold) by exploring all possible pairs
        int_max = np.int64(np.iinfo(np.int64).max)
        best_feature_index, best_threshold, minimum_cost = (
            int_max,
            np.float64(np.inf),
            np.float64(np.inf),
        )
        best_split: list[Dataset] | None = None
        idx: np.int64
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            val: np.float64 = np.random.choice(values)
            left_dataset, right_dataset = dataset.split(idx, val)
            cost = self._CART_cost(left_dataset, right_dataset)   # J(k,v)
            if cost < minimum_cost:
                (
                    best_feature_index,
                    best_threshold,
                    minimum_cost,
                    best_split,
                ) = (idx, val, cost, [left_dataset, right_dataset])

        if best_split == None or best_feature_index == np.inf:
            raise Exception('Dataset is empty')

        return best_feature_index, best_threshold, minimum_cost, best_split
