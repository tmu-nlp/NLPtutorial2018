import sys
sys.path.append('..')
from utils import nn
from utils import data
import numpy as np

if __name__ == '__main__':
        F = nn.FeedForwardLayer
        S = nn.Sigmoid
        T = nn.Tanh
        layers = ((F, 100, S), (F, None, T))
        bow = data.S1DataSet('../../data/titles-en-train.labeled', 10, True)
        opt = nn.SteepestGradientOptimizer(1, 0.4)
        w = nn.Network(layers, bow, opt)
        opt.init_weights(np.random.uniform)
        print(w)
        w.train(30)
