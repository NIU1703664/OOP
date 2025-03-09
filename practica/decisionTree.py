from typing import override, Self


class Node:
    def predict(self, inputs: list[float]) -> int:
        pass
class Leaf(Node):
    def __init__(self, label: int) -> None:
        super().__init__()
        self.label: int= label
    @override
    def predict(self, inputs:list[float]) -> int:
        return self.label

class Parent(Node):
    def __init__(self, k_index:int, v:float) -> None:
        super().__init__()
        self.feature_index: int = k_index
        self.threshold: float = v
        self.left_child: Self
        self.right_child: Self
    @override
    def predict(self, inputs: list[float]) -> int:
        if inputs[self.feature_index] < self.threshold:
            return self.left_child.predict(inputs)
        else:
            return self.right_child.predict(inputs)



