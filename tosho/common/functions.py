import math
import numpy as np

def softmax(x):
    if x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x)
        x /= np.sum(x)
    elif x.ndim == 2:
        # バッチ処理
        # 各データ(axis=1)内で最大値を取得して、他のデータから減算する.
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)

    return x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7

    # yの各ベクトルについて、正解ラベルのインデックスにある要素を取り出す
    a = y[np.arange(batch_size), t]

    return -1 * np.sum(np.log(a + delta)) / batch_size

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