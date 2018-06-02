# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from collections import Counter
from multiprocessing import Pool
from utils.n_gram import interpolate_gen, n_gram
from functools import partial
from random import shuffle
open = partial(open, encoding='UTF-8')

class Perceptron:
    def __init__(self, in_dim, weights = None):
        self._weights = np.zeros(in_dim)

    def predict(self, x, t = None):
        result = np.matmul(self._weights, x)
        y = np.sign(result) #.flatten()
        if t is not None: # or gradient
            return y != t
        return y

    def _update(self, x_t):
        x, t = x_t
        err_idx = self.predict(x, t)
        loss = np.average(err_idx)
        if loss == 0:
            return 0, np.zeros_like(t)
        x = x.T[err_idx].T
        t = t[err_idx]
        return loss, np.sum(t * x, axis = 1)

    def update(self, x, t):
        mis, deltas = self._update((x, t))
        self._weights += deltas
        return mis

    def update_mp(self, xs, ts, pool):
        n = len(xs)
        r_ob = pool.map_async(self._update, zip(xs, ts))
        r_ob.wait()
        avg_mis = sum(mis for mis, _ in r_ob.get()) / n
        deltas = np.sum(deltas for _, deltas in r_ob.get())
        self._weights += deltas
        return avg_mis

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, w):
        if w.shape != self._weights.shape:
            raise TypeError("Bad size")
        self._weights = w

def _vectorize(features, inputs, bias):
    num_dim = len(features)
    num_sen = len(inputs)
    vectors = np.zeros((num_dim + 1 if bias else num_dim, num_sen))
    for i, sentence in enumerate(inputs):
        for tok, cnt in sentence.items():
            if tok not in features:
                continue
            vectors[features[tok], i] = cnt
    if bias:
        vectors[num_dim, :] += bias
    return vectors

class Trainer:
    def __init__(self, name, feature_maker = lambda x:Counter(x.strip().split())+Counter(n_gram(2, x.strip().split()))):
        self._perceptron = None
        self._features = set()
        self._labels = []
        self._inputs = []
        self._name = name
        self._bias = None
        self._featurize = feature_maker

    def add_corpus(self, fname):
        with open(fname) as fr:
            for line in fr:
                label, sentence = line.split('\t') # tokenize & feature extraction
                self._labels.append(int(label))
                features = self._featurize(sentence)
                self._inputs.append(features)
                self._features |= set(features)

    def seal(self, bias = None):
        if self._features:
            self._features = {feat:pos for pos, feat in enumerate(self._features)}
            n = len(self._features)
            if bias:
                self._bias = bias
                n += 1
            self._perceptron = Perceptron( n )
            # temp.shape = temp.shape + (1,)
        else:
            w, self._features = self.np.load(self._name + '.npz')
            self._perceptron = Perceptron(len(self._features), w)

    def train(self, num_epoch = 30, stop_loss = 0.1, batch_size = 100, mp = 8, train_set_ratio = 0.9):
        p = self._perceptron
        interpolate = interpolate_gen(0.9)
        total = len(self._inputs)
        idx = list(range(total))
        shuffle(idx)
        train_set_ratio = int(total * train_set_ratio)
        train_set_idx = idx[:train_set_ratio]
        valid_set_idx = idx[train_set_ratio:]
        validation_x = tuple(self._inputs[i] for i in valid_set_idx)
        validation_y = tuple(self._labels[i] for i in valid_set_idx)
        vloss = '-'
        blend_loss = None
        best_weights = (1, None)
        # in order to apply multi-threading, update should return the deltas to
        # the main thread to sum up (map-reduce) the deltas.
        if mp > 1:
            print('mp =', mp)
            pool = Pool(processes = mp)
        for i in range(num_epoch):
            t = 0
            while t < train_set_ratio:
                if mp > 1:
                    inputs = []
                    labels = []
                    for _ in range(mp):
                        rand_inputs = [self._inputs[i] for i in train_set_idx[t:t + batch_size]]
                        rand_labels = [self._labels[i] for i in train_set_idx[t:t + batch_size]]
                        inputs.append(_vectorize(self._features, rand_inputs, self._bias))
                        labels.append(np.asarray(rand_labels))
                        t += batch_size
                        if t >= train_set_ratio:
                            break
                    tloss = p.update_mp(inputs, labels, pool)
                else:
                    inputs = _vectorize(self._features, train_x[t:t + batch_size], self._bias)
                    labels = np.asarray(train_y[t:t + batch_size])
                    t += batch_size
                    tloss = p.update(inputs, labels)

                if valid_set_idx:
                    inputs = _vectorize(self._features, validation_x, self._bias)
                    labels = np.asarray(validation_y)
                    vloss = np.average(p.predict(inputs, labels))
                    if blend_loss is None:
                        blend_loss = vloss
                    else:
                        blend_loss = interpolate(vloss, blend_loss)
                print(f"Error rate epoch.{i}({t}/{train_set_ratio}|{total}): {tloss} {vloss}")
                shuffle(train_set_idx)
            if isinstance(blend_loss, float):
                if blend_loss < best_weights[0]:
                    best_weights = (blend_loss, p.weights.copy())
                    np.savez(self._name, best_weights[1], self._features)
                if blend_loss < stop_loss:
                    print('reached the goal, blended: ', blend_loss)
                    break  # more selective
        p.weights = best_weights[1]
        print("Train end")

    def csv(self, fname):
        with open(fname, 'w') as fw:
            fw.write('tok,weight\n')
            for tok, pos in self._features.items():
                fw.write(f'{tok},{self._perceptron.weights[pos]}\n')
            if self._bias:
                fw.write(f'#BIAS#,{self._perceptron.weights[len(self._features)]}')


    def test(self, fname):
        sentences = []
        with open(fname) as fr:
            for line in fr:
                sentences.append(Counter(line.split()))
        inputs = _vectorize(self._features, sentences, True)
        return self._perceptron.predict(inputs)

    def __str__(self):
        s = 'A perceptron wrapper\n'
        s += '\tfeatures size: %d\n' % len(self._features)
        s += '\tCorpus size: %d\n' % len(self._inputs)
        return s

if __name__ == '__main__':
    w = Trainer("data_titles")
    w.add_corpus('../../data/titles-en-train.labeled')
    #w.add_corpus('../../test/03-train-input.txt')
    w.seal(bias = 5)
    print(w)
    w.train()
    res = w.test('../../data/titles-en-test.word')
    with open("my.labels", "w") as fw:
        for i in res:
            fw.write('%d\n' % i)
    w.csv('weights.csv')
