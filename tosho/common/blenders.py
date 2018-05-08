
from collections import defaultdict

class SimpleBlender:
    '''
    常に一定
    '''
    def __init__(self, unk_rate=0.05):
        self.unk = unk_rate

    def unk_rate(self, *words):
        return self.unk

class MultiLayerBlender:
    '''
    n-gramごとの補完係数
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
    def unk_rate(self, *words):
        pass