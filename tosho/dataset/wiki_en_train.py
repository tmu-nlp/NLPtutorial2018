import sys, os
sys.path.append(os.pardir)
import dataset.utils as utils

def load_norm_pos(file_name='../../data/wiki-en-train.norm_pos'):
    X, T = [], []
    X.append([])
    T.append([])
    for x, t in utils.load_norm_pos(file_name):
        if x == '</s>':
            X.append([])
            T.append([])
        else:
            X[-1].append(x)
            T[-1].append(t)
    # 最後の空の配列を除外する
    return X[:-1], T[:-1]