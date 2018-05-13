
import os, sys
sys.path.append(os.path.pardir)
from common.utils import iterate_tokens, count_tokens
from collections import defaultdict

class SimpleSmoothing:
    '''
    n-gramによらず常に一定
    '''
    def __init__(self, unk_rate=0.05):
        self.unk = unk_rate

    def unk_rate(self, *words):
        return self.unk

class MultiLayerSmoothing:
    '''
    n-gram別に補完係数を設定する
    '''
    def __init__(self, default_unk_rate=0.05, unk_rates={}):
        self.unk_rates = defaultdict(lambda : default_unk_rate)
        for n, rate in unk_rates.items():
            self.unk_rates[n] = rate
        print(self.unk_rates)
    
    def unk_rate(self, *words):
        return self.unk_rates[len(words)]

class WittenBell:
    '''
    Witten Bell 平滑化
    '''
    def __init__(self, t_data, n_gram=1):
        self.unk_rates = {}
        self.base_unk_rate = None

        for n in range(n_gram):
            self.__init_n(t_data, n)
    
    def __init_n(self, t_data, n):
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
            
            rates = defaultdict(lambda : 1.0)
            for key in word_u.keys():
                rates[key] = word_u[key] / (word_c[key] + word_u[key])

            self.unk_rates[n] = rates
        
    def unk_rate(self, *words):
        if len(words) == 0:
            return self.base_unk_rate
        else:
            return self.unk_rates[len(words)][words]

if __name__ == '__main__':
    with open('../../test/02-train-input.txt', 'r') as f:
        wb = WittenBell(list(f), 3)

    for n, rate in wb.unk_rates.items():
        for w, r in rate.items():
            print(f'l{w} = {r}')
    print(f'l(unk)={wb.base_unk_rate}')
    for words in [('a'), ('a', 'b'), ('z')]:
        print(f'l({words}) = {wb.unk_rate(*words)}')