from .randomForest import Forest
import numpy as np
import sys
import logging

def main():
    logging.info("Starting the program")
    if len(sys.argv) == 0:
        logging.error("This program requires an argument")
        return
        
    dataset = sys.argv[0];
    
    y: list[str] =[]
    dX: list[list[float]]=[[]]
    if dataset == "lily": #???
        #Include importted database 
        print("Important");
    else:
        f = open(f"./dataset/${dataset}.csv", 'r')
        y: list[str] =[]
        dX: list[list[float]]=[[]]
        
        for line in f.readlines():
            fields = line.split(',')
            y.append(fields[len(fields)])
            try:
                dX.append([float(param) for param in fields[:-1]])
            except:
                logging.error("Couldn't read file, check formatting"

        f.close()
    
    # Hyperparameters
    num_trees = 100 # number of decision trees
    max_depth = 10 # maximum number of levels of a decision tree
    min_size_split = 5 # if less, do not split a node
    ratio_samples = 0.7 # sampling with replacement
    num_random_features = int(np.sqrt(num_features))
    criterion = 'gini'
    
    logging.info("Creating the Random Forest")
    forest = Forest(num_trees, max_depth, min_size_split, ratio_samples, num_random_features, criterion)
    logging.info("Random Forest created")
