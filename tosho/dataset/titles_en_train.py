import sys, os
sys.path.append(os.pardir)
import dataset.utils as utils

def load_labeled(file_name='../../data/titles-en-train.labeled'):
    X, T = [], []
    for x, t in utils.load_labeled(file_name):
        X.append(x)
        T.append(t)
    
    return X, T