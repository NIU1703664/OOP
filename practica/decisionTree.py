from __future__ import annotations
from abc import ABC, abstractmethod
from typing import override
import numpy as np
import numpy.typing as npt


class Node(ABC):
    @abstractmethod
    def predict(self, row: npt.NDArray[np.float64]) -> np.int64:
        pass

    @abstractmethod
    def accept(self, visitor: NodeVisitor):
        pass


class Leaf(Node):
    def __init__(self, label: np.int64) -> None:
        super().__init__()
        self.label: np.int64 = label
        # logging.info(f'Created a leaf node with an etiquete: {label}')

    @override
    def predict(self, row: npt.NDArray[np.float64]) -> np.int64:
        return self.label

    @override
    def accept(self, visitor: NodeVisitor):
        visitor.visitLeaf(self)


class Parent(Node):
    def __init__(self, k_index: np.int64, v: np.float64) -> None:
        super().__init__()
        self.feature_index: np.int64 = k_index
        self.threshold: np.float64 = v
        self.left_child: Node
        self.right_child: Node
        # logging.info(
        #     f'Created a node with father with a feature_index: {k_index}, threshold: {v}'
        # )

    @override
    def predict(self, row: npt.NDArray[np.float64]) -> np.int64:
        assert row.ndim == 1
        if row[self.feature_index] < self.threshold:
            return self.left_child.predict(row)
        else:
            return self.right_child.predict(row)

    @override
    def accept(self, visitor: NodeVisitor):
        visitor.visitParent(self)
        self.left_child.accept(visitor)
        self.right_child.accept(visitor)


class NodeVisitor(ABC):
    @abstractmethod
    def visitLeaf(self, node: Leaf):
        pass

    @abstractmethod
    def visitParent(self, node: Parent):
        pass


class PrintNode(NodeVisitor):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.depth: int = depth

    @override
    def visitLeaf(self, node: Leaf):
        print(self.depth, f'leaf, {node.label}')

    @override
    def visitParent(self, node: Parent):
        print(
            self.depth,
            f'parent - {node.feature_index}, {node.threshold}',
        )
        self.depth += 1
        node.left_child.accept(self)
        node.left_child.accept(self)
