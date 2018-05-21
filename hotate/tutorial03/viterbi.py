# -*- coding: utf-8 -*-

import math
from collections import defaultdict


def load_model(path):
    prob = defaultdict(lambda: 0)
    for line in open(path, 'r'):
        line = line.split()
        prob[line[0]] = float(line[1])
    return prob


def viterbi(path):
    prob = load_model('model_file')
    lam = 0.95
    V = 1000000
    for line in open(path, 'r'):
        best_edge = defaultdict()
        best_score = defaultdict()
        best_edge[0] = None
        best_score[0] = 0
        for end_num in range(1, len(line)):
            best_score[end_num] = 10 ** 10
            for begin_num in range(len(line)):
                if begin_num == end_num:
                    break
                else:
                    word = line[begin_num:end_num]
                    if word in prob.keys() or len(word) == 1:
                        probability = lam * prob[word] + (1 - lam) / V
                        my_score = best_score[begin_num] + -math.log(probability)
                        if my_score < best_score[end_num]:
                            best_score[end_num] = my_score
                            best_edge[end_num] = [begin_num, end_num]

        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge is not None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        words.reverse()
        sentence = ' '.join(words)
        yield sentence


if __name__ == '__main__':
    # path = '../../test/04-input.txt'
    # prob = load_model('../../test/04-model.txt')
    path = '../../data/wiki-ja-test.txt'
    with open('my_answer.word', 'w') as f:
        for sentence in viterbi(path):
            f.write(f'{sentence}\n')

'''
Sent Accuracy: 23.81% (20/84)
Word Prec: 71.88% (1943/2703)
Word Rec: 84.22% (1943/2307)
F-meas: 77.56%
Bound Accuracy: 86.30% (2784/3226)
'''