from abc import abstractmethod
from typing import override, Generic
import numpy as np
from .dataset import Label
import logging
logging.basicCongif(level=logging.INFO)

class Node(Generic[Label]):
    @abstractmethod
    def predict(self, inputs: list[float]) -> Label:
        pass

class Leaf(Node[Label]):
    def __init__(self, label: Label) -> None:
        super().__init__()
        self.label: Label = label
        logging.info(f"Created a leaf node with an etiquete: {label}")
    @override
    def predict(self, inputs:list[float]) -> Label:
        return self.label

class Parent(Node[Label]):
    def __init__(self, k_index: np.int64, v:np.float64) -> None:
        super().__init__()
        self.feature_index: np.int64 = k_index
        self.threshold: np.float64 = v
        self.left_child: Node[Label]
        self.right_child: Node[Label]
        logging.info(f"Created a node with father with a feature_index: {k_index}, threshold: {v}"
    @override
    def predict(self, inputs: list[float]) -> Label:
        if inputs[self.feature_index] < self.threshold:
            return self.left_child.predict(inputs)
        else:
            return self.right_child.predict(inputs)



