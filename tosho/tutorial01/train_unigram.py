import os, sys
sys.path.append(os.path.pardir)
from tutorial00.word_count import count_words, load_file
from random import sample
import math
import pickle

class UniGram:
    def __init__(self, verbose=False):
        self.words = None       # 単語ごとの確率
        self.unk = None         # 未知語の確率
        self.entropy = None     # エントロピー
        self.perplexity = None  # パープレキシティ
        self.coverage = None    # カバレッジ

        self.verbose = verbose

    def train(self, word_dict, vocab_size=10**6, unk_rate=0.05):
        # 未知語確率を計算する
        self.unk = unk_rate / vocab_size
        
        # 単語ごとの確率を計算する
        nk_rate = 1. - unk_rate
        N = sum(word_dict.values())
        self.words = dict(word_dict)
        for key, value in self.words.items():
            self.words[key] = nk_rate * value / N + self.unk

        if self.verbose:
            self.__print_params()

    def __print_params(self):
        print(f'Unknow Rate : {self.unk}')
        print(f'Trained words (sample) :')
        for word, rate in sample(self.words.items(), min(10, len(self.words))):
            print(f'{word} : {rate}')

    def load(self, filename):
        with open(filename, 'rb') as fp:
            self.words, self.unk = pickle.load(fp)

        if self.verbose:
            self.__print_params()

    def save(self, filename):
        with open(filename, 'wb') as fp:
            pickle.dump((self.words, self.unk), fp)

    def estimate(self, seq):
        entropy = 0.
        W = 0
        unks = 0
        for sentence in seq:
            for word in sentence.split(' '):
                if word in self.words:
                    entropy += math.log(self.words[word], 2)
                else:
                    entropy += math.log(self.unk, 2)
                    # 未知語の数をカウントする（カバレッジ用）
                    unks += 1
                W += 1
                if self.verbose:
                    print(f'{W} : {word}, entropy={entropy}')
        
        self.entropy = -1 * entropy / W
        self.coverage = (W - unks) / W
        self.perplexity = math.pow(2, self.entropy)

        return self.entropy, self.coverage, self.perplexity
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_train_file')
    parser.add_argument('path_to_cache_file')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--includes-eos', action='store_true', default=False)
    parser.add_argument('--verify-params', action='store_true', default=False)
    arg = parser.parse_args()

    print('training model...')

    train_file = arg.path_to_train_file

    words = count_words(train_file, includes_eos=arg.includes_eos)
    model = UniGram(arg.debug)
    model.train(words)

    model.save(arg.path_to_cache_file)

    if arg.verify_params:
        model2 = UniGram(arg.debug)
        model2.load(arg.path_to_cache_file)

        print(f'UNK | actual : {model.unk}, saved : {model2.unk}')
        for key in sample(model.words.keys(), min(len(model.words), 10)):
            print(f'{key} | actual : {model.words[key]}, saved : {model2.words[key]}')