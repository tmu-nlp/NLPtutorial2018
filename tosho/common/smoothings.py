
import os, sys
sys.path.append(os.path.pardir)
from common.utils import iterate_tokens, count_tokens
from collections import defaultdict
import pickle

class SimpleSmoothing:
    '''
    n-gramによらず常に一定
    '''
    def __init__(self, unk_rate=0.05):
        self.unk = unk_rate

    def unk_rate(self, *words):
        return self.unk

    def save_params(self, file_name = 'smoothing.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(('simple', self.unk), f)
    
    def load_params(self, file_name = 'smoothing.pkl'):
        with open(file_name, 'rb') as f:
            name, self.unk = pickle.load(f)

class MultiLayerSmoothing:
    '''
    n-gram別に補完係数を設定する
    '''
    def __init__(self, unk_rates={}):
        self.unk_rates = defaultdict(float)
        for n, rate in unk_rates.items():
            self.unk_rates[n] = rate
    
    def unk_rate(self, *words):
        return self.unk_rates[len(words)]

    def save_params(self, file_name = 'smoothing.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(('multilayer', self.unk_rates), f)
    
    def load_params(self, file_name = 'smoothing.pkl'):
        with open(file_name, 'rb') as f:
            name, self.unk_rates = pickle.load(f)

class WittenBell:
    '''
    Witten Bell 平滑化
    '''
    def __init__(self, n_gram=1):
        self.n_gram = n_gram
        self.unk_rates = {}
        self.base_unk_rate = None

    def train(self, t_data):
        for n in range(self.n_gram):
            if n == 0:
                counts = count_tokens(iterate_tokens(t_data, 1))
                c = sum(counts.values())
                u = len(counts)
                self.base_unk_rate = u / (c + u)
            else:
                # c(w)
                word_c = count_tokens(iterate_tokens(t_data, n))

                # u(w)
                two_grams = count_tokens(iterate_tokens(t_data, n+1))
                word_u = defaultdict(int)
                for pair in two_grams.keys():
                    key = pair[0:-1]
                    word_u[key] += 1
                
                rates = defaultdict(self.unseen_unk_rate)
                for key in word_u.keys():
                    rates[key] = word_u[key] / (word_c[key] + word_u[key])

                self.unk_rates[n] = rates
        
    def unk_rate(self, *words):
        key = words[1:]
        if len(key) == 0:
            return self.base_unk_rate
        else:
            return self.unk_rates[len(key)][key]

    def unseen_unk_rate(self):
        return 1.0

    def save_params(self, file_name = 'smoothing.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(('witten-bell', self.n_gram, self.unk_rates, self.base_unk_rate), f)
    
    def load_params(self, file_name = 'smoothing.pkl'):
        with open(file_name, 'rb') as f:
            name, self.n_gram, self.unk_rates, self.base_unk_rate = pickle.load(f)

if __name__ == '__main__':
    with open('../../test/02-train-input.txt', 'r') as f:
        wb = WittenBell(3)
        wb.train(list(f))

    for n, rate in wb.unk_rates.items():
        for w, r in rate.items():
            print(f'l{w} = {r}')
    print(f'l(unk)={wb.base_unk_rate}')
    for words in [('a'), ('a', 'b'), ('z')]:
        print(f'l({words}) = {wb.unk_rate(*words)}')