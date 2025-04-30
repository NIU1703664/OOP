#!/bin/python3
from randomForest import Forest, RandomForest, ExtraTrees
from dataset import Dataset
from measure import Impurity, Gini, Entropy
import numpy as np
import numpy.typing as npt
import sys
import logging
import argparse

# Hyperparameters
num_trees: int = 84   # number of decision trees
max_depth: int = 20   # maximum number of levels of a decision tree
min_size_split: int = 5   # if less, do not split a node
ratio_samples: float = 0.8   # sampling with replacement
ratio_train = 0.7


def benchmark(forest: Forest, dataset: Dataset):
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
                args.paralel,
            )
        case 'extra':
            forest = ExtraTrees(
                num_trees,
                max_depth,
                min_size_split,
                ratio_samples,
                num_random_features,
                criterion,
                args.paralel,
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
    print('                   Sequential    |     Paralel   ')
    print('                -----------------|-----------------')
    print(
        f'Random Forest     {sr_time:2.3f}s, {(100*sr_acc):2.1f}%  |  {pr_time:2.3f}s, {(100*pr_acc):2.1f}% '
    )
    print(
        f'Extra Trees       {se_time:2.3f}s, {(100*se_acc):2.1f}%  |  {pe_time:2.3f}s, {(100*pe_acc):2.1f}% '
    )


def main():
    # add argument
    parser = argparse.ArgumentParser(description='Random Forest classifier')
    _ = parser.add_argument(
        'dataset',
        help='Which dataset to train and test',
    )
    _ = parser.add_argument(
        '-m',
        '--measure',
        default='gini',
        help='Select purity measure algorithm',
    )
    _ = parser.add_argument(
        '-p',
        '--paralel',
        nargs='?',
        const=True,
        default=False,
        help='Build the forest with parallel processing',
    )
    _ = parser.add_argument(
        '-a',
        '--arch',
        default='random',
        help='Tree architecture',
    )
    _ = parser.add_argument(
        '--full_benchmark',
        nargs='?',
        const=True,
        default=False,
        help='Compare in a table all parallelization and architecture options, overrides --paralel and --arch',
    )
    args = parser.parse_args()
    if args.dataset == None:
        print('This program requires an argument')
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
    if args.full_benchmark:
        test_all(args, dataset)
    else:
        test_single(args, dataset)


if __name__ == '__main__':
    main()
