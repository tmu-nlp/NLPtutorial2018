XOR = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([1, 1], [0])
]

DATA = {
    'XOR': XOR
}

from random import sample
import numpy as np

def load_data(operant='XOR', data_size=1000):
    '''
    論理回路の学習データを出力する
    '''
    seed_data = DATA[operant]

    X, T = [], []
    for _ in range(data_size):
        x, t = sample(seed_data, 1)[0]
        
        X.append(x)
        T.append(t)
    
    return np.array(X), np.array(T)

if __name__ == '__main__':
    x, t = load_data(data_size=10)

    for pair in zip(x, t):
        print(*pair)