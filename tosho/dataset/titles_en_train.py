import sys, os
sys.path.append(os.pardir)
from common.utils import load_labeled_data

def load_labeled(file_name='../../data/titles-en-train.labeled'):
    X, T = [], []
    for x, t in load_labeled_data(file_name):
        X.append(x)
        T.append(t)
    
    return X, T