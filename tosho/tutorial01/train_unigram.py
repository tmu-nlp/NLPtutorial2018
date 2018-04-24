import os, sys
sys.path.append(os.path.pardir)
from tutorial00.word_count import count_words

class UniGram:
    def __init__(self):
        self.words = None   # 単語ごとの確率
        self.unk = None # 未知語の確率

    def train(self, word_dict, vocab_size=10**6, unk_rate=0.05):
        # 未知語確率を計算する
        self.unk = unk_rate / vocab_size
        
        # 単語ごとの確率を計算する
        nk_rate = 1. - unk_rate
        N = sum(word_dict.values())
        self.words = word_dict
        for key, value in self.words.items():
            self.words[key] = nk_rate * value / N + self.unk

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def entropy(self, sentences):
        pass

    def coverage(self, sentences):
        pass
    
if __name__ == '__main__':
    source_filename = sys.argv[1]
    word_dict = count_words(source_filename)

    model = UniGram()
    model.train(word_dict)

    print(model.unk)
    print(model.words.items())