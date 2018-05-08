import os, sys
sys.path.append(os.path.pardir)
from common.utils import count_words, parse_file
import math
import pickle

class ZeroGram:
    '''
    p(unk)の値を返すためのクラス。
    補完係数はここでは考慮しないので、1/語彙数を返す。
    '''
    def __init__(self):
        self.unk = None
    
    def train(self, vocab_size=10**6):
        self.unk = 1 / vocab_size
    
    def estimate(self, *words):
        return self.unk

    def save_cache(self, cache):
        pickle.dump((self.unk), cache)
    
    def load_cache(self, cache):
        self.unk = pickle.load(cache)

    def print_params(self):
        print(f'p(unk) = {self.unk}')

class NGram:
    '''
    N-Gramでエントロピーを計算するためのクラス。
    '''
    def __init__(self, n_gram=1):
        self.n = n_gram                 # n-gram
        self.words = None               # 学習した条件付き確率
        self.unk_rate = None            # 未知語率
        self.n_minus_one_gram = None    # (n-1)-gram. 出現確率を計算するときに使用する

    def train(self, train_file, vocab_size=10**6, unk_rate=0.05):
        # unigramの(n-1)-gramにはZeroGramクラスを使用する
        # これは estimate で 1/vocab_size を常に返すクラスである
        if self.n == 1:
            self.n_minus_one_gram = ZeroGram()
            self.n_minus_one_gram.train(vocab_size=vocab_size)
        else:
            self.n_minus_one_gram = NGram(self.n - 1)
            self.n_minus_one_gram.train(train_file, vocab_size=vocab_size, unk_rate=unk_rate)
        
        # 各n-gramの確率を計算する
        self.words = count_words(train_file, self.n)
        total_tokens = sum(self.words.values())
        for key, count in self.words.items():
            self.words[key] = count / total_tokens
        
        # 保管係数を計算する
        self.unk_rate = unk_rate

    def estimate(self, *words):
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
        p_n_1 = self.n_minus_one_gram.estimate(*sub_words)
        
        # 未知語率を考慮して確率を計算する
        p = (1. - self.unk_rate) * p_n + self.unk_rate * p_n_1
        # print(f'{words} | (1. - {self.unk_rate}) * {p_n} + {self.unk_rate} * {p_n_1} = {p}')

        return p

    def entropy(self, test_filename):
        entropy = 0.
        W = 0

        with open(test_filename, 'r') as test_file:
            doc = parse_file(test_file, self.n)
            for seq in doc:
                for pair in seq:
                    p = self.estimate(*pair)
                    entropy += math.log(p, 2)
                    W += 1
        
        return -1 * entropy / W
    
    def save(self, cache_filename):
        '''
        依存関係にあるモデルも含めて１つのファイルに保存する
        '''
        with open(cache_filename, 'wb') as cache:
            self.save_cache(cache)
    
    def save_cache(self, cache):
        # 自身のモデルを保存する
        pickle.dump((self.n, self.words, self.unk_rate), cache)
        # (n-1)-gram のモデルを保存する
        self.n_minus_one_gram.save_cache(cache)
    
    def load(self, cache_filename):
        with open(cache_filename, 'rb') as cache:
            self.load_cache(cache)

    def load_cache(self, cache):
        self.n, self.words, self.unk_rate = pickle.load(cache)
        
        if self.n == 1:
            self.n_minus_one_gram = ZeroGram()
        else:
            self.n_minus_one_gram = NGram(self.n - 1)
        
        self.n_minus_one_gram.load_cache(cache)
    
    def print_params(self):
        print(f'{len(self.words)} types ({self.n}-gram)')
        for key, value in sorted(self.words.items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f'p({key}) = {value}')
        
        self.n_minus_one_gram.print_params()

'''
p(b|a) = l * p(b|a) + (1 - l) * p(b)
p(b) = m * p(b) + (1 - m) * p(unk)

words['2gram']
{
    p(b|a) : 0.8
    p(c|a) : 0.2
}
words['1gram']
{
    p(a) : 0.7
    p(b) : 0.2
    p(c) : 0.1
}
words['0gram']
{
    p(unk) = 1.0
}


train model with ../../data/wiki-en-train.word
5234 words
p(unk) = 5.0000000000000004e-08
p(,) = 0.04720584208749511
p(the) = 0.03830008906032029
p(</s>) = 0.03448333776295966
p(.) = 0.03397973863344679
p(of) = 0.029738903858601638
p(a) = 0.021601802134367503
p(to) = 0.019958478659115007
p(and) = 0.01834166040120529
p(in) = 0.0141538360610457
p(is) = 0.013040616932648847
saved parameters to wiki-en-train.pyc
'''