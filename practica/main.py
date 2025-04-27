import time
from randomForest import Forest
from dataset import Dataset
from measure import Impurity, Gini, Entropy
import numpy as np
import numpy.typing as npt
import sys
import logging


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
        case _:
            print('Dataset not found, try another option: ')
            print('- sonar')
            print('- iris')
            return

    logging.info('Creating the Random Forest')
    # Hyperparameters
    num_trees: int = 100   # number of decision trees
    criterion: Impurity = Entropy()
    max_depth: int = 10   # maximum number of levels of a decision tree
    min_size_split: int = 5   # if less, do not split a node
    ratio_samples: float = 0.7   # sampling with replacement
    num_random_features: int = int(
        np.sqrt(dataset.num_features)
    )   # This number is not chosen at random but represents the number of features to choose at random
    forest = Forest(
        num_trees,
        max_depth,
        min_size_split,
        ratio_samples,
        num_random_features,
        criterion,
    )
    logging.info('Random Forest created')

    logging.info('Fitting Forest to dataset')
    logging.info('Time starts now!')
    t1 = time.time()
    ratio_train = 0.7
    num_samples_train: int = int(dataset.num_samples * ratio_train)
    num_samples_test: int = dataset.num_samples - num_samples_train
    idx = np.random.permutation(range(dataset.num_samples))
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train : num_samples_train + num_samples_test]
    X_train, y_train = dataset.X[idx_train], dataset.y[idx_train]
    X_test, y_test = dataset.X[idx_test], dataset.y[idx_test]
    forest.fit(X_train, y_train)
    t2 = time.time()
    logging.info(f'Training time: {t2-t1}s')

    logging.info('Classifying elements in the test class')
    ypred: npt.NDArray[np.int64] = forest.predict(X_test)

    logging.info('Calculating accuracy')
    # logging.info(ypred)
    # logging.info(y_test)
    hits: int = np.sum(ypred == y_test)
    accuracy: float = hits / float(num_samples_test)
    print(f'Accuracy {100*np.round(accuracy,decimals=2)} %')


main()
