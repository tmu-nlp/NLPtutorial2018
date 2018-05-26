# -*- coding: utf-8 -*-
import math
from collections import defaultdict

class DevideWords:
    def __init__(self):
        self.lambda_1 = 0.95
        self.lambda_unk = round(1 - self.lambda_1, 2)
        self.V = 10**6
        self.unigram = defaultdict(int)

    def import_model(self, filename, deliminator=','):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                pair = line.split(deliminator)
                self.unigram[pair[0]] = float(pair[1])

    def viterbi_forward(self, _line):
        best_scores = [0]
        best_edges = [None]
        for word_end in range(1, len(_line) + 1):
            best_score = 10**10
            best_edge = None
            for word_begin in range(0, word_end):
                word = _line[word_begin:word_end]
                if word in self.unigram.keys() or len(word) == 1:
                    prob = self.lambda_1 * self.unigram[word] + self.lambda_unk / self.V
                    my_score = best_scores[word_begin] - math.log2(prob)
                    if my_score < best_score:
                        best_score = my_score
                        best_edge = (word_begin, word_end)
            best_scores.append(best_score)
            best_edges.append(best_edge)

        return best_scores, best_edges

    def viterbi_backward(self, _line, _best_edges):
        words = []
        next_edge = _best_edges[len(_best_edges) - 1]
        while next_edge is not None:
            word = _line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = _best_edges[next_edge[0]]
        words.reverse()
        return words

