# -*- coding: utf-8 -*-

import math
from collections import defaultdict


def bigram_test(path):
    probs = defaultdict(lambda: 0)
    lam = defaultdict(lambda: 0)
    V = 10 ** 6
    # l1 = 0.95
    H = 0
    W = 0
    for line in open('model_file', 'r'):
        model = line.strip().split('\t')
        probs[model[0]] = float(model[1])
        lam[model[0]] = float(model[2])

    for line in open(path, 'r'):
        words = line.strip().split(' ')
        words.append('</s>')
        words.insert(0, '<s>')
        for i in range(1, len(words)):
            # p1 = (l1 * probs[words[i]]) + ((1-l1) / V)
            p1 = (lam[words[i]] * probs[words[i]]) + ((1-lam[words[i]]) / V)
            p2 = (lam[words[i-1] + ' ' + words[i]] * probs[words[i-1] + ' ' + words[i]]) + ((1-lam[words[i-1] + ' ' + words[i]]) * p1)
            H += -math.log2(p2)
            W += 1

    entropy = H / W
    print(f'entropy = {entropy}')


if __name__ == '__main__':
    path = '../../data/wiki-en-test.word'

    bigram_test(path)


