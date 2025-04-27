from abc import abstractmethod
from typing import override
import numpy as np
import numpy.typing as npt
import logging

logging.basicConfig(level=logging.INFO)


class Node:
    @abstractmethod
    def predict(self, row: npt.NDArray[np.float64]) -> np.int64:
        pass


class Leaf(Node):
    def __init__(self, label: np.int64) -> None:
        super().__init__()
        self.label: np.int64 = label
        # logging.info(f'Created a leaf node with an etiquete: {label}')

    @override
    def predict(self, row: npt.NDArray[np.float64]) -> np.int64:
        return self.label


class Parent(Node):
    def __init__(self, k_index: np.int64, v: np.float64) -> None:
        super().__init__()
        self.feature_index: np.int64 = k_index
        self.threshold: np.float64 = v
        self.left_child: Node
        self.right_child: Node
        logging.info(
            f'Created a node with father with a feature_index: {k_index}, threshold: {v}'
        )

    @override
    def predict(self, row: npt.NDArray[np.float64]) -> np.int64:
        assert row.ndim == 2
        if row[self.feature_index] < self.threshold:
            return self.left_child.predict(row)
        else:
            return self.right_child.predict(row)
