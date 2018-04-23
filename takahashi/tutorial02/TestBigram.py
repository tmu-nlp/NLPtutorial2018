# -*- coding: utf-8 -*-
import math
import re


class TestBigram():
    def __init__(self):
        self.probs = {}

    def import_model(self, model_file):
        with open(model_file, encoding='utf-8')as f:
            for line in f:
                line = re.sub(r'[\r\n]', '', line)
                pair = line.split(',')
                self.probs[pair[0]] = float(pair[1])
        print('keys:{0}'.format(len(self.probs.keys())))

    def evaluate(self, test_file):
        lambda_1 = 0.95
        lambda_2 = 0.95
        V = 1000000
        W = 0
        H = 0

        with open(test_file, encoding='utf-8')as f:
            for line in f:
                line = line.lower()
                line = re.sub(r'[\.,\^\?!\+-;\'\"`~=\[\]\{\}\$%&\\\*]', '', line)
                line = re.sub(r'[\r\n]', '', line)
                line = re.sub(r' +', ' ', line)
                words = line.split(' ')
                words.insert(0, '<s>')
                words.append('</s>')
                for index in range(1, len(words)):
                    if words[index] == '' or words[index - 1] == '':
                        continue

                    P1 = lambda_1 * self.probs[words[index]] + round(1 - lambda_1, 2) / V
                    P2 = lambda_2 * self.probs[words[index - 1] + ' ' + words[index]] + round(1 - lambda_2, 2) * P1

                    H += -math.log2(P2)
                    W += 1
        entropy = round(H/float(W), 6)

        print('entropy = {}'.format(entropy))

    def witten_bell_smoothing(self, word, context_word):
        lambda_w = 1 - self.probs[0]


if __name__ == '__main__':
    tester = TestBigram()

    tester.import_model('test_model.txt')
    tester.evaluate('../../test/02-train-input.txt')
