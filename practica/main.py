#!/bin/python3
# To be able to execute the file standalone
from ForestLib.randomForest import Classifier, Forest, Regressor
from ForestLib.dataset import Dataset
from ForestLib.measure import Impurity, Gini, Entropy, SSE
from ForestLib.splitting import Split, ExtraSplit, RandomSplit
import numpy as np
import numpy.typing as npt
import logging
import argparse
import argcomplete

# Hyperparameters small
# num_trees: int = 84   # number of decision trees
# max_depth: int = 20   # maximum number of levels of a decision tree
# min_size_split: int = 5   # if less, do not split a node
# ratio_samples: float = 0.8   # sampling with replacement
# ratio_train = 0.7

# Hyperparameters big
num_trees: int = 42  # number of decision trees
max_depth: int = 20   # maximum number of levels of a decision tree
min_size_split: int = 20   # if less, do not split a node
ratio_samples: float = 0.4   # sampling with replacement
ratio_train = 0.7

# Hyperparameters debug
# num_trees: int = 1  # number of decision trees
# max_depth: int = 20   # maximum number of levels of a decision tree
# min_size_split: int = 20   # if less, do not split a node
# ratio_samples: float = 0.4   # sampling with replacement
# ratio_train = 0.7


def benchmark(
    forest: Forest, dataset: Dataset, verbose: bool
) -> tuple[float, str]:
    num_samples_train: int = int(dataset.num_samples * ratio_train)
    num_samples_test: int = dataset.num_samples - num_samples_train
    idx = np.random.permutation(range(dataset.num_samples))
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train : num_samples_train + num_samples_test]
    X_train, y_train = dataset.X[idx_train], dataset.y[idx_train]
    X_test, y_test = dataset.X[idx_test], dataset.y[idx_test]

    forest.fit(X_train, y_train)
    ypred: npt.NDArray[np.int64] | npt.NDArray[np.float64] = forest.predict(
        X_test
    )
    if verbose:
        forest.print_trees()
        forest.featureGraph(dataset.mnist)

    if type(forest) == Regressor:
        assert type(ypred[0]) == np.float64
        accuracy: float = np.sqrt(
            np.sum((ypred - y_test) ** 2) / num_samples_test
        )
        result_str: str = f'{(accuracy):2.1f} rmse'

    else:
        assert type(ypred[0]) == np.int64
        hits: int = np.sum(ypred == y_test)
        accuracy: float = hits / float(num_samples_test)
        result_str: str = f'{(100*accuracy):2.1f}%'
    return (forest.time, result_str)


def test_single(
    forest: type[Forest],
    criterion: Impurity,
    arch: str,
    parallel: bool,
    dataset: Dataset,
    print_trees: bool,
):
    num_random_features: int = int(
        np.sqrt(dataset.num_features)
    )   # This number is not chosen at random but represents the number of features to choose at random
    split: Split
    match arch:
        case 'random':
            split = RandomSplit(criterion)
        case 'extra':
            split = ExtraSplit(criterion)
        case _:
            print('Invalid architecture selected')
            return

    time, acc = benchmark(
        forest(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            split,
            parallel,
        ),
        dataset,
        print_trees,
    )
    print(f'Time: {time:2.3f}s, Accuracy: {acc}')


def test_all(
    forest: type[Forest],
    measure: Impurity,
    dataset: Dataset,
    print_trees: bool,
):
    num_random_features: int = int(
        np.sqrt(dataset.num_features)
    )   # This number is not
    extra_split, random_split = ExtraSplit(measure), RandomSplit(measure)
    sr_time, sr_acc = benchmark(
        forest(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            random_split,
            False,
        ),
        dataset,
        print_trees,
    )
    pr_time, pr_acc = benchmark(
        forest(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            random_split,
            True,
        ),
        dataset,
        print_trees,
    )
    se_time, se_acc = benchmark(
        forest(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            extra_split,
            False,
        ),
        dataset,
        print_trees,
    )
    pe_time, pe_acc = benchmark(
        forest(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            extra_split,
            True,
        ),
        dataset,
        print_trees,
    )
    print('')
    print('                   Sequential    |     parallel   ')
    print('                -----------------|-----------------')
    print(
        f'Random forest     {sr_time:2.3f}s, {sr_acc}  |  {pr_time:2.3f}s, {pr_acc}'
    )
    print(
        f'Extra Trees       {se_time:2.3f}s, {se_acc}  |  {pe_time:2.3f}s, {pe_acc}'
    )


def main(
    dataset: str,
    arch: str,
    measure: str,
    full_benchmark: bool,
    log_level: str,
    parallel: bool,
    print_trees: bool,
):
    log_levels = {
        'INFO': logging.INFO,
        'WARN': logging.WARN,
        'DEBUG': logging.DEBUG,
    }
    logging.basicConfig(
        level=log_levels[log_level],
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logging.info('Starting the program')
    logging.info(f'Attemtping to load {dataset}:')
    data: Dataset
    forest: type[Forest] = Classifier
    match dataset:
        case 'sonar':
            data = Dataset.load_sonar()
            logging.info(f'Sonar database loaded')
        case 'iris':
            data = Dataset.load_iris()
            logging.info(f'Iris database loaded')
        case 'mnist':
            data = Dataset.load_MNIST()
            logging.info(f'MNIST database loaded')
        case 'temperatures':
            data = Dataset.load_temperatures()
            forest = Regressor
            measure = 'sse'
            logging.info(f'Min_Temperatures database loaded')
        case _:
            return
    logging.info(f'{dataset} loaded')
    criterion: Impurity
    match measure:
        case 'gini':
            criterion = Gini()
        case 'entropy':
            criterion = Entropy()
        case 'sse':
            criterion = SSE()
        case _:
            return

    if full_benchmark:
        logging.info(f'Starting a complete benchmark')
        test_all(forest, criterion, data, print_trees)
    else:
        logging.info(f'Testing {arch} with {parallel} computing')
        test_single(forest, criterion, arch, parallel, data, print_trees)


if __name__ == '__main__':
    # CLI Commands
    parser = argparse.ArgumentParser(
        description='Random Forest classifier',
        usage='./main.py [options...] dataset',
    )
    _ = parser.add_argument(
        'dataset',
        help='Which dataset to train and test',
        choices=['iris', 'sonar', 'mnist', 'temperatures'],
    )
    _ = parser.add_argument(
        '-m',
        '--measure',
        default='gini',
        help='Select purity measure algorithm',
        choices=['gini', 'entropy', 'sse'],
    )
    _ = parser.add_argument(
        '-a',
        '--arch',
        default='random',
        help='Tree architecture',
        choices=['random', 'extra'],
    )
    _ = parser.add_argument(
        '-p',
        '--parallel',
        action='store_true',
        help='Build the forest with parallel processing',
    )
    _ = parser.add_argument(
        '--full_benchmark',
        action='store_true',
        help='Compare in a table all parallelization and architecture options, overrides --parallel and --arch',
    )
    _ = parser.add_argument(
        '--print_trees',
        action='store_true',
        help='Compare in a table all parallelization and architecture options, overrides --parallel and --arch',
    )
    _ = parser.add_argument(
        '--log_level',
        default='WARN',
        help='Set log level',
        choices=['INFO', 'DEBUG', 'WARNING'],
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    main(**vars(args))
