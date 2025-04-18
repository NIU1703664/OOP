import numpy as np
from typing import Generic
from numpy.typing import NDArray
from .decisionTree import Leaf, Parent, Node
from .dataset import Dataset, Label
from .measure import Impurity
import logging

class Forest (Generic[Label]):
    def __init__(self, num_trees: int, min_size: int, max_depth: int, ratio_samples: float, num_random_features: int, criterion: str) -> None:
        self.num_trees: int = num_trees
        self.min_size: int = min_size
        self.max_depth: int = max_depth
        self.ratio_samples: float = ratio_samples
        self.num_random_features: int = num_random_features
        self.criterion: str = criterion
        self.decision_trees: list[Node[Label]] = []
        logging.info(f"Starting a Random Forest with {num_trees} trees")

    def predict(self, x: list[float]) -> Label:
        logging.debug(f"Making prediction for the entrance: {x}")
        labels: dict[Label, int] = dict()
        max_value = 0
        max_label: Label | None = None
        for tree in self.decision_trees:
            predict: Label = tree.predict(x)
            if predict in labels:
                labels[predict] += 1
            else:
                labels[predict] = 1

            if labels[predict] > max_value:
                max_value = labels[predict]
                max_label = predict

        if max_label == None:
            logging.error("Forest hasn''t been fit yet")
            raise Exception("Forest hasn''t been fit yet")

        return max_label


    def fit(self, X: NDArray[np.float64], y: NDArray[Label]):
# a pair (X,y) is a dataset, with its own responsibilities
        logging.info("Starting the training of the Random Forest")
        dataset = Dataset(X,y)
        self._make_decision_trees(dataset)
        logging.info("Training finished")
        

    def _make_decision_trees(self, dataset: Dataset[Label]):
        self.decision_trees = []
        logging.info(f"making {self.num_trees} decision trees")
        for i in range(self.num_trees):
# sample a subset of the dataset with replacement using
# np.random.choice() to get the indices of rows in X and y
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset,1) # the root of the decision tree
            self.decision_trees.append(tree)
            logging.debug(f"Tree {i+1} created")


    def _make_node(self, dataset: Dataset[Label], depth: int) -> Node[Label]:
        logging.debug(f"Creating node in depth {depth} with {dataset.num_samples} samples")
        if depth == self.max_depth or dataset.num_samples <= self.min_size or len(np.unique(dataset.y)) == 1:
# last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        return node


    def _make_leaf(self, dataset: Dataset[Label]) -> Leaf[Label]:
        # label = most frequent class in dataset
        label = dataset.most_frequent_label()
        logging.info(f"Creating a leaf with label {label}")
        return Leaf(label)


    def _make_parent_or_leaf(self, dataset: Dataset[Label], depth: int) -> Node[Label]:
# select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.num_features), self.num_random_features, replace=False)

        best_feature_index, best_threshold, minimum_cost, best_split = self._best_split(idx_features, dataset)

        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
# this is an special case : dataset has samples of at least two
# classes but the best split is moving all samples to the left or right
# dataset and none to the other, so we make a leaf instead of a parent
            return self._make_leaf(dataset)
        else:
            node = Parent[Label](best_feature_index, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
        return node


    def _best_split(self, idx_features: NDArray[np.int64], dataset: Dataset[Label]) -> tuple[np.int64, np.float64, np.float64, list[Dataset[Label]]]: # find the best pair (feature, threshold) by exploring all possible pairs
        int_max = np.int64(np.iinfo(np.int64).max)
        best_feature_index, best_threshold, minimum_cost = int_max, np.float64(np.inf), np.float64(np.inf)
        best_split: list[Dataset[Label]] | None = None
        idx: np.int64
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            val: np.float64
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset) # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, best_split = idx, val, cost, [left_dataset, right_dataset]

        if best_split == None or best_feature_index == np.inf:
            raise Exception("Dataset[Label] is empty")

        return best_feature_index, best_threshold, minimum_cost, best_split


    def _CART_cost(self, left_dataset: Dataset[Label], right_dataset: Dataset[Label]) -> np.float64:
    # the J(k,v) equation in the slides, using Gini
        left_len = np.float64(left_dataset.num_samples)
        right_len = np.float64(right_dataset.num_samples)
        critereon: Impurity = self.criterion

        return left_len / (left_len + right_len) * critereon.purity(left_dataset) + right_len / (left_len + right_len) * critereon.purity(right_dataset)
