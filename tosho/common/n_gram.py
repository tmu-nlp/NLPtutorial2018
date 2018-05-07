import os, sys
sys.path.append(os.path.pardir)
from common.utils import count_words
from collections import defaultdict
import math
import pickle

class ZeroGram:
    '''
    p(unk)の値を返すためのクラス
    '''
    def __init__(self):
        self.unk = None
    
    def train(self, vocab_size=10**6, unk_rate=0.05):
        self.unk = unk_rate / vocab_size
    
    def estimate(self, *words):
        return self.unk

class NGram:
    def __init__(self, n_gram=1):
        self.n = n_gram
        
        self.words = defaultdict(int)
        self.unk_rate = None

        self.n_minus_one_gram = None

    def train(self, train_file, vocab_size=10**6, unk_rate=0.05):
        if self.n == 1:
            self.n_minus_one_gram = ZeroGram()
            self.n_minus_one_gram.train(vocab_size=vocab_size, unk_rate=unk_rate)
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
        p_n = self.words[words]
        return (1. - self.unk_rate) * p_n + self.unk_rate * self.n_minus_one_gram.estimate(words[1:])

    # def entropy(self, test_filename):
    #     entropy = 0.
    #     W = 0
    #     unks = 0
        
    #     with open(test_filename, 'r') as test_file:
    #         doc = parse_file(test_file, self.includes_eos)
    #         for line in doc:
    #             for word in line.split(' '):
    #                 if word in self.words:
    #                     entropy += math.log(self.words[word], 2)
    #                 else:
    #                     entropy += math.log(self.unk, 2)
    #                     # 未知語の数をカウントする（カバレッジ用）
    #                     unks += 1
    #                 W += 1
        
    #     entropy = -1 * entropy / W
    #     coverage = (W - unks) / W
    #     perplexity = math.pow(2, entropy)

    #     return entropy, coverage, perplexity


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