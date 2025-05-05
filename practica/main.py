#!/bin/python3
# To be able to execute the file standalone
from randomForest import Forest, RandomForest, ExtraTrees
from dataset import Dataset
from measure import Impurity, Gini, Entropy
import numpy as np
import numpy.typing as npt
import logging
import argparse

# Hyperparameters
num_trees: int = 84   # number of decision trees
max_depth: int = 20   # maximum number of levels of a decision tree
min_size_split: int = 5   # if less, do not split a node
ratio_samples: float = 0.8   # sampling with replacement
ratio_train = 0.7


def benchmark(forest: Forest, dataset: Dataset) -> tuple[float, float]:
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
    return (forest.time, accuracy)


def test_single(args, dataset: Dataset):
    num_random_features: int = int(
        np.sqrt(dataset.num_features)
    )   # This number is not chosen at random but represents the number of features to choose at random
    forest: Forest
    criterion: Impurity
    match args.measure:
        case 'gini':
            criterion = Gini()
        case 'entropy':
            criterion = Entropy()
        case _:
            print('Invalid measure selected')
            return
    match args.arch:
        case 'random':
            forest = RandomForest(
                num_trees,
                max_depth,
                min_size_split,
                ratio_samples,
                num_random_features,
                criterion,
                args.parallel,
            )
        case 'extra':
            forest = ExtraTrees(
                num_trees,
                max_depth,
                min_size_split,
                ratio_samples,
                num_random_features,
                criterion,
                args.parallel,
            )
        case _:
            print('Invalid architecture selected')
            return
    time, acc = benchmark(forest, dataset)
    print(f'Time: {time:2.3f}s, Accuracy: {(100*acc):2.1f}%')


def test_all(args, dataset: Dataset):
    num_random_features: int = int(
        np.sqrt(dataset.num_features)
    )   # This number is not chosen at random but represents the number of features to choose at random
    criterion: Impurity
    match args.measure:
        case 'gini':
            criterion = Gini()
        case 'entropy':
            criterion = Entropy()
        case _:
            print('Invalid measure selected')
            return
    sr_time, sr_acc = benchmark(
        RandomForest(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            criterion,
            False,
        ),
        dataset,
    )
    pr_time, pr_acc = benchmark(
        RandomForest(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            criterion,
            True,
        ),
        dataset,
    )
    se_time, se_acc = benchmark(
        ExtraTrees(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            criterion,
            False,
        ),
        dataset,
    )
    pe_time, pe_acc = benchmark(
        ExtraTrees(
            num_trees,
            max_depth,
            min_size_split,
            ratio_samples,
            num_random_features,
            criterion,
            True,
        ),
        dataset,
    )
    print('')
    print('                   Sequential    |     parallel   ')
    print('                -----------------|-----------------')
    print(
        f'Random Forest     {sr_time:2.3f}s, {(100*sr_acc):2.1f}%  |  {pr_time:2.3f}s, {(100*pr_acc):2.1f}% '
    )
    print(
        f'Extra Trees       {se_time:2.3f}s, {(100*se_acc):2.1f}%  |  {pe_time:2.3f}s, {(100*pe_acc):2.1f}% '
    )


def main():
    # CLI Commands
    parser = argparse.ArgumentParser(
        description='Random Forest classifier',
        usage='./main.py [options...] dataset',
    )
    _ = parser.add_argument(
        'dataset',
        help='Which dataset to train and test (iris, sonar, mnist)',
    )
    _ = parser.add_argument(
        '-m',
        '--measure',
        default='gini',
        help='Select purity measure algorithm (gini, entropy)',
    )
    _ = parser.add_argument(
        '-a',
        '--arch',
        default='random',
        help='Tree architecture (random, extra)',
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
        '--log_level',
        default='WARN',
        help='Set log level',
    )
    args = parser.parse_args()
    log_levels = {
        'INFO': logging.INFO,
        'WARN': logging.WARN,
        'DEBUG': logging.DEBUG,
    }
    logging.basicConfig(
        level=log_levels[args.log_level],
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    if args.dataset == None:
        print(
            'This program requires an argument, use -h or --help for information on how to use this program.'
        )
        return

    logging.info('Starting the program')
    logging.info(f'Attemtping to load {args.dataset}:')
    dataset: Dataset
    match args.dataset:
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
    logging.info(f'{args.dataset} loaded')
    if args.full_benchmark:
        logging.info(f'Starting a complete benchmark')
        test_all(args, dataset)
    else:
        logging.info(f'Testing {args.arch} with {args.parallel} computing')
        test_single(args, dataset)


if __name__ == '__main__':
    main()
