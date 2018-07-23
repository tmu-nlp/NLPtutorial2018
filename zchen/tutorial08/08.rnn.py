import sys
sys.path.append('..')
from utils import nn
from utils import data
import numpy as np

if __name__ == '__main__':
        F = nn.FeedForwardLayer
        R = nn.RecurrentLayer
        S = nn.Sigmoid
        T = nn.Tanh
        layers = ((R, 100, S), (F, None, T))
        bow = data.SSDataSet("../../data/wiki-en-test.norm", 10, True)
        opt = nn.SteepestGradientOptimizer(1, 0.4)
        w = nn.Network(layers, bow, opt)
        opt.init_weights(np.random.uniform)
        print(w)
        w.train(30)
