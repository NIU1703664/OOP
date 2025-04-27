from randomForest import Forest
from dataset import Dataset, Label
from measure import Gini, Impurity
import numpy as np
import numpy.typing as npt
import sys
import logging



def main():
    logging.info('Starting the program')
    if len(sys.argv) == 0:
        print('This program requires an argument')
        return

    dataset = Dataset(np.array([]), np.array([]))
    print(f'Attemtping to load {sys.argv[1]}:')
    match sys.argv[1]:
        case 'lilly':
            if dataset.load_lilly:
                logging.info('Lily dataset loaded!')
            else: 
                logging.error('Could not load Lilly, see logs')
        case 'sonar':
            if dataset.load_sonar:
                logging.info('Sonar dataset loaded!')
            else: 
                logging.error('Could not load Sonar, see logs')
        case 'iris':
            if dataset.load_iris:
                logging.info('Iris dataset loaded!')
            else: 
                logging.error('Could not load Lilly, see logs')
        case _:
            print('Dataset not found, try another option: ')
            print('- lilly')
            print('- sonar')
            print('- iris')
            return

    logging.info('Creating the Random Forest')
    # Hyperparameters
    num_trees: int = 100   # number of decision trees
    criterion: Impurity = Gini()
    max_depth: int = 10   # maximum number of levels of a decision tree
    min_size_split: int = 5   # if less, do not split a node
    ratio_samples = 0.7   # sampling with replacement
    num_random_features = int(np.sqrt(dataset.num_features))
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
    ratio_train = 0.7
    num_samples_train: int = int(dataset.num_samples*ratio_train)
    num_samples_test: int= dataset.num_samples-num_samples_train
    idx = np.random.permutation(range(dataset.num_samples))
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train : num_samples_train+num_samples_test]
    X_train, y_train = dataset.X[idx_train], dataset.y[idx_train]
    X_test, y_test = dataset.X[idx_test], dataset.y[idx_test]
    forest.fit(X_train, y_train)

    logging.info('Classifying elements in the test class')
    ypred: npt.NDArray = forest.predict(X_test)

    logging.info('Calculating accuracy')
    hits: float = np.sum(ypred == y_test)
    accuracy: float = hits/float(num_samples_test)
    print(f'Accuracy {100*np.round(accuracy,decimals=2)} %')


main()
