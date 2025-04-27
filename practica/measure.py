from abc import abstractmethod
import numpy as np
from dataset import Dataset
from typing import override

# import logging


class Impurity:
    @abstractmethod
    def purity(self, dataset: Dataset) -> np.float64:
        pass


class Gini(Impurity):
    @override
    def purity(self, dataset: Dataset) -> np.float64:
        counts = np.bincount(dataset.y)
        probs = counts / dataset.num_samples
        return 1 - np.sum(probs**2)


class Entropy(Impurity):
    @override
    def purity(self, dataset: Dataset) -> np.float64:
        counts = np.bincount(dataset.y)
        probs = counts / dataset.num_samples
        return -np.sum(probs * np.log(probs))
