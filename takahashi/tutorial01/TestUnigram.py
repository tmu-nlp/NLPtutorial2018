# -*- coding: utf-8 -*-
import math
import re


class TestUnigram():
    def __init__(self):
        self.probabilities = {}

    def import_model(self, model_file):
        with open(model_file, encoding='utf-8')as f:
            for line in f:
                line = re.sub(r'[\r\n]', '', line)
                pair = line.split(',')
                self.probabilities[pair[0]] = float(pair[1])
        print('keys:{0}'.format(len(self.probabilities.keys())))

    def evaluate(self, test_file):
        lambda_1 = 0.95
        lambda_unk = round(1 - lambda_1, 2)
        V = 1000000
        W = 0
        H = 0

        unk = 0

        with open(test_file, encoding='utf-8')as f:
            for line in f:
                line = re.sub(r'[\r\n]', '', line)
                words = line.split(' ')
                words.append('</s>')
                for word in words:
                    W += 1
                    P = lambda_unk / V
                    if word in self.probabilities:
                        P += lambda_1 * self.probabilities[word]
                    else:
                        unk += 1
                    H += -math.log2(P)
        entropy = round(H/float(W), 6)
        coverage = round((W - unk)/float(W), 6)


        print('entropy = {}'.format(entropy))
        print('coverage = {}'.format(coverage))

if __name__ == '__main__':
    tester = TestUnigram()

    tester.import_model('model.txt')
    tester.evaluate('../../test/01-test-input.txt')
