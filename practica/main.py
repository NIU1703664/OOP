from randomForest import Forest, RandomForest, ExtraTrees
from dataset import Dataset
from measure import Impurity, Gini, Entropy
import numpy as np
import numpy.typing as npt
import sys
import logging


def benchmark(forest: Forest, dataset: Dataset):
    ratio_train = 0.7
    num_samples_train: int = int(dataset.num_samples * ratio_train)
    num_samples_test: int = dataset.num_samples - num_samples_train
    idx = np.random.permutation(range(dataset.num_samples))
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train : num_samples_train + num_samples_test]
    X_train, y_train = dataset.X[idx_train], dataset.y[idx_train]
    X_test, y_test = dataset.X[idx_test], dataset.y[idx_test]

    forest.fit(X_train, y_train)
    ypred: npt.NDArray[np.int64] = forest.predict(X_test)
    hits: int = np.sum(ypred == y_test)
    accuracy: float = hits / float(num_samples_test)

    return f'{np.round(forest.time,decimals=3):5}s  {100*np.round(accuracy,decimals=2):3}%'


def main():
    logging.info('Starting the program')
    if len(sys.argv) < 2:
        print('This program requires an argument')
        return

    logging.info(f'Attemtping to load {sys.argv[1]}:')
    dataset: Dataset
    match sys.argv[1]:
        case 'sonar':
            dataset = Dataset.load_sonar()
            logging.info(f'Sonar database loaded')
        case 'iris':
            dataset = Dataset.load_iris()
            logging.info(f'Iris database loaded')
        case 'mnist':
            dataset = Dataset.load_MNIST()
            logging.info(f'MNIST database loaded')
        case _:
            print('Dataset not found, try another option: ')
            print('- sonar')
            print('- iris')
            print('- mnist')
            return

    logging.info('Creating the Random Forest')
    # Hyperparameters
    num_trees: int = 120   # number of decision trees
    criterion: Impurity = Gini()
    max_depth: int = 10   # maximum number of levels of a decision tree
    min_size_split: int = 5   # if less, do not split a node
    ratio_samples: float = 0.7   # sampling with replacement
    num_random_features: int = int(
        np.sqrt(dataset.num_features)
    )   # This number is not chosen at random but represents the number of features to choose at random
    s_random = RandomForest(
        num_trees,
        max_depth,
        min_size_split,
        ratio_samples,
        num_random_features,
        criterion,
        False,
    )
    p_random = RandomForest(
        num_trees,
        max_depth,
        min_size_split,
        ratio_samples,
        num_random_features,
        criterion,
        True,
    )
    s_extra = ExtraTrees(
        num_trees,
        max_depth,
        min_size_split,
        ratio_samples,
        num_random_features,
        criterion,
        False,
    )
    p_extra = ExtraTrees(
        num_trees,
        max_depth,
        min_size_split,
        ratio_samples,
        num_random_features,
        criterion,
        True,
    )
    sr_result = benchmark(s_random, dataset)
    pr_result = benchmark(p_random, dataset)
    se_result = benchmark(s_extra, dataset)
    pe_result = benchmark(p_extra, dataset)
    print()
    print('                   Sequential    |     Paralel   ')
    print('                -----------------|-----------------')
    print(f'Random Forest     {sr_result}  |  {pr_result}')
    print(f'Extra Trees       {se_result}  |  {pe_result}')


main()
