from random import random
import numpy as np

PREDICATOR ={
    'AND': lambda a, b: 1 if a>0 and b>0 else 0,
    'OR':  lambda a, b: 0 if a<0 and b<0 else 1,
    'XOR': lambda a, b: 0 if a*b>0 else 1
}

def load_data(operant='XOR', data_size=1000):
    '''
    論理回路の学習データを出力する
    '''
    pred = PREDICATOR[operant]

    X, T = [], []
    for _ in range(data_size):
        x = random() * 2 - 1
        y = random() * 2 - 1
        t = pred(x, y)
        
        X.append([x, y])
        T.append(t)
    
    return np.array(X), np.array(T)

if __name__ == '__main__':
    x, t = load_data(data_size=10)

    for pair in zip(x, t):
        print(*pair)