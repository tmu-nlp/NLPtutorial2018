import math
import numpy as np

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c) # avoid overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7

    # yの各ベクトルについて、正解ラベルのインデックスにある要素を取り出す
    biggests_in_y = y[np.arange(batch_size), t]

    return -1 * np.sum(np.log(biggests_in_y + delta)) / batch_size

def sigmoid(x):
    exp = math.exp(x)
    return exp / (1 + exp)

def d_sigmoid(x):
    exp = math.exp(x)
    return exp / (1 + exp)**2

if __name__ == '__main__':
    inp = [-10, -5, -1]

    for i in inp:
        print(sigmoid(i))