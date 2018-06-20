import os, sys
sys.path.append(os.path.pardir)
from tutorial00.word_count import count_words, parse_file
from random import sample
import math
import pickle

class UniGram:
    def __init__(self, includes_eos=False):
        self.words = None       # 単語ごとの確率
        self.unk = None         # 未知語の確率

        self.includes_eos = includes_eos

    def train(self, train_file, vocab_size=10**6, unk_rate=0.05):
        word_dict = count_words(train_file, includes_eos=self.includes_eos)
        
        # 未知語確率を計算する
        self.unk = unk_rate / vocab_size
        
        # 単語ごとの確率を計算する
        nk_rate = 1. - unk_rate
        N = sum(word_dict.values())
        self.words = dict(word_dict)
        for key, value in self.words.items():
            self.words[key] = nk_rate * value / N + self.unk

    def load(self, cache_file):
        with open(cache_file, 'rb') as fp:
            self.words, self.unk, self.includes_eos = pickle.load(fp)

    def save(self, cache_file):
        with open(cache_file, 'wb') as fp:
            pickle.dump((self.words, self.unk, self.includes_eos), fp)

    def estimate(self, test_filename):
        entropy = 0.
        W = 0
        unks = 0
        
        with open(test_filename, 'r') as test_file:
            doc = parse_file(test_file, self.includes_eos)
            for line in doc:
                for word in line.split(' '):
                    if word in self.words:
                        entropy += math.log(self.words[word], 2)
                    else:
                        entropy += math.log(self.unk, 2)
                        # 未知語の数をカウントする（カバレッジ用）
                        unks += 1
                    W += 1
        
        entropy = -1 * entropy / W
        coverage = (W - unks) / W
        perplexity = math.pow(2, entropy)

        return entropy, coverage, perplexity

    def print_params(self):
        print(f'{len(self.words)} words')
        print(f'p(unk) = {model.unk}')
        for key, value in sorted(self.words.items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f'p({key}) = {value}')
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_train_file')
    parser.add_argument('path_to_cache_file')
    parser.add_argument('--includes-eos', action='store_true', default=False)
    arg = parser.parse_args()

    print(f'train model with {arg.path_to_train_file}')

    model = UniGram(arg.includes_eos)
    model.train(arg.path_to_train_file)
    
    model.print_params()

    model.save(arg.path_to_cache_file)

    print(f'saved parameters to {arg.path_to_cache_file}')

'''
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