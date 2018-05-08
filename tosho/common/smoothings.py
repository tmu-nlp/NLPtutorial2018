
import os, sys
sys.path.append(os.path.pardir)
from common.utils import count_words
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
    def __init__(self, train_filename):
        # c(w)
        self.word_c = count_words(train_filename, 1)

        # u(w)
        two_grams = count_words(train_filename, 2)
        self.word_u = defaultdict(int)
        for pair in two_grams.keys():
            key = pair[0:1]
            self.word_u[key] += 1
        
    def unk_rate(self, *words):
        key = words[-1:]
        
        c = self.word_c[key]
        u = self.word_u[key]

        # 未知語の場合
        if c == 0 and u == 0:
            return 1
        else:
            return u / (c + u)