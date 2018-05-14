import os, sys
sys.path.append(os.path.pardir)
from common.utils import count_words, parse_file, count_tokens, iterate_tokens
import math
import pickle
from collections import defaultdict

class ZeroGram:
    '''
    p(unk)の値を返すためのクラス。
    補完係数はここでは考慮しないので、1/語彙数を返す。
    '''
    def __init__(self):
        self.unk = None
    
    def set_smoothing(self, smoothing):
        pass

    def train(self, vocab_size=10**6):
        self.unk = 1 / vocab_size
    
    def prob(self, *words):
        return self.unk

    def get_params(self):
        params = {}
        params[0] = self.unk
        return params
    
    def set_params(self, params):
        self.unk = params[0]

    def print_params(self):
        print(f'p(unk) = {self.unk}')

class NGram:
    '''
    N-Gramでエントロピーを計算するためのクラス。
    '''
    def __init__(self, n_gram=1):
        self.n = n_gram                 # n-gram
        self.words = None               # 学習した条件付き確率
        self.n_minus_one_gram = None    # (n-1)-gram. 出現確率を計算するときに使用する
        self.smoothing = None           # 平滑化アルゴリズム（未知語率を算出する）

    def set_smoothing(self, smoothing):
        self.smoothing = smoothing
        self.n_minus_one_gram.set_smoothing(smoothing)

    def train(self, t_data, vocab_size=10**6):
        '''
        t_data : sequence of string
        '''
        # 各n-gramの確率を計算する
        self.words = count_tokens(iterate_tokens(t_data, self.n))
        sub_totals = defaultdict(int)
        for key, count in self.words.items():
            sub_key = key[:-1]
            sub_totals[sub_key] += count

        for key, count in self.words.items():
            self.words[key] = count / sub_totals[key[:-1]]
            
        # unigramの(n-1)-gramにはZeroGramクラスを使用する
        # これは estimate で 1/vocab_size を常に返すクラスである
        if self.n == 1:
            self.n_minus_one_gram = ZeroGram()
            self.n_minus_one_gram.train(vocab_size=vocab_size)
        else:
            self.n_minus_one_gram = NGram(self.n - 1)
            self.n_minus_one_gram.train(t_data, vocab_size=vocab_size)

    def prob(self, *words):
        '''
        Parameters
        =====
        words : 文中の順番で配列になっていることを想定する。
                A cat sat ... で trigram であれば、['A', 'cat', 'sat'] となる。
        '''
        # 学習した確率
        p_n = self.words[words]

        # 補完に使用する確率を求める
        sub_words = words[1:]
        p_n_1 = self.n_minus_one_gram.prob(*sub_words)

        # 補完係数を求める
        unk_rate = self.smoothing.unk_rate(*words)
        
        # 未知語率を考慮して確率を計算する
        p = (1. - unk_rate) * p_n + unk_rate * p_n_1
        # print(f'{words} | (1. - {unk_rate}) * {p_n} + {unk_rate} * {p_n_1} = {p}')

        return p

    def entropy(self, t_data):
        H = 0.
        W = 0

        for token in iterate_tokens(t_data, self.n):
            p = self.prob(*token)
            H += math.log2(p)
            W += 1
        
        return -1 * H / W
    
    def save_params(self, file_name='params.pkl'):
        '''
        依存関係にあるモデルも含めて１つのファイルに保存する
        '''
        params = self.get_params()
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        
    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        
        self.set_params(params)
    
    def get_params(self):
        params = self.n_minus_one_gram.get_params()

        params[self.n] = self.words
        return params

    def set_params(self, params):
        self.n = max(params.keys())
        self.words = params.pop(self.n)

        # (n-1)-gram のパラメータを設定する
        if self.n == 1:
            self.n_minus_one_gram = ZeroGram()
        else:
            self.n_minus_one_gram = NGram(self.n - 1)
        self.n_minus_one_gram.set_params(params)
    
    def print_params(self):
        print(f'{len(self.words)} types ({self.n}-gram)')
        for key, value in sorted(self.words.items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f'p({key}) = {value}')
        
        self.n_minus_one_gram.print_params()
