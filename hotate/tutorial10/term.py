# -*- coding: utf-8 -*-
import math
from collections import defaultdict


class Preterm:
    def __init__(self):
        self.sym = []
        self.word = []
        self.prob = []

    def __call__(self, sym, word, prob):
        self.sym.append(sym)
        self.word.append(word)
        self.prob.append(-math.log(float(prob)))

    def index(self, word):
        for i, pre_word in enumerate(self.word):
            if pre_word == word:
                yield self.sym[i], self.prob[i]


class Nonterm:
    def __init__(self):
        self.sym = []
        self.lsym = []
        self.rsym = []
        self.prob = []

    def __call__(self, sym, lsym, rsym, prob):
        self.sym.append(sym)
        self.lsym.append(lsym)
        self.rsym.append(rsym)
        self.prob.append(-math.log(float(prob)))

    def all(self):
        for i in range(len(self.sym)):
            yield self.sym[i], self.lsym[i], self.rsym[i], self.prob[i]

    def update(self, best, r, l, m):
        for i in range(len(self.sym)):
            lsym = f'{self.lsym[i]} {l} {m}'
            rsym = f'{self.rsym[i]} {m} {r}'
            if lsym in best.score and rsym in best.score:
                my_lp = best.score[lsym] + best.score[rsym] + self.prob[i]
                if my_lp < best.score[f'{self.sym[i]} {l} {r}']:
                    best.score[f'{self.sym[i]} {l} {r}'] = my_lp
                    best.edge[f'{self.sym[i]} {l} {r}'] = (lsym, rsym)


class Best:
    def __init__(self):
        self.score = defaultdict(lambda: 10**6)
        self.edge = {}

    def scores(self, l, r, sym, prob):
        self.score[f'{sym} {l} {r}'] = prob
