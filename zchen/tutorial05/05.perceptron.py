# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from collections import Counter
from multiprocessing import Pool
from utils.n_gram import interpolate_gen, n_gram
from functools import partial
from random import shuffle
from stemming.porter2 import stem
open = partial(open, encoding='UTF-8')

class Perceptron:
    def __init__(self, in_dim, weights = None):
        if weights is None:
            print(f'New Perceptron {in_dim}')
            self._weights = np.zeros(in_dim)
            self._update_sd = np.zeros(in_dim)
        else:
            self._weights = weights
            self._update_sd = np.zeros_like(weights)
        #self._itp = interpolate_gen(0.5)

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
        return loss, np.sum(t * x, axis = 1) # 1 x(feat, batch) get rid of var in 1

    def update(self, x, t):
        itp = interpolate_gen(0.5)
        loss, deltas = self._update((x, t))
        self._weights += deltas
        self._update_sd = itp(deltas ** 2, self._update_sd)
        return loss

    def update_mp(self, xs, ts, pool):
        n = len(xs)
        itp = interpolate_gen(0.5)
        r_ob = pool.map(self._update, zip(xs, ts))
        self._weights += np.sum(deltas for _, deltas in r_ob)
        self._update_sd = itp(np.sum(deltas**2 for _, deltas in r_ob)/n, self._update_sd)
        return sum(loss for loss, _ in r_ob) / n

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, w):
        if w.shape != self._weights.shape:
            raise TypeError("Bad size")
        self._weights = w

    @property
    def convergency(self):
        return self._update_sd


def _vectorize(feat2id, inputs, bias):
    num_dim = len(feat2id)
    num_sen = len(inputs)
    vectors = np.zeros((num_dim + 1 if bias else num_dim, num_sen))
    for i, sentence in enumerate(inputs):
        for tok, cnt in sentence.items():
            if tok not in feat2id:
                continue
            vectors[feat2id[tok], i] = cnt
    if bias:
        vectors[num_dim, :] += bias
    # normalize # get rid of dim v in #0 (feats)
    # vectors = np.log(vectors + np.e )i # not flex
    vectors /= np.sqrt(np.sum(vectors ** 2, axis = 0)) # no neg
    return vectors

class Trainer:
    def __init__(self, name, feature_maker = lambda x:Counter(x.strip().split())):
        self._perceptron = None
        self._features = Counter()
        self._labels = []
        self._inputs = []
        self._name = name
        self._featurize = feature_maker

    def add_corpus(self, fname):
        with open(fname) as fr:
            for line in fr:
                label, sentence = line.split('\t') # tokenize & feature extraction
                self._labels.append(int(label))
                features = self._featurize(sentence)
                self._inputs.append(features)
                self._features += features

    def train(self, num_epoch = 30, stop_loss = 0.03, batch_size = 100, mp = 8, train_set_ratio = 0.9, bias = 1, decrease_threshold = 0.25):
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
        selected_features = self._features
        loss_info = 'epoch,tloss,vloss\n'
        convergency_info = 'epoch,tok,mu,std\n'
        move = True
        # in order to apply multi-threading, update should return the deltas to
        # the main thread to sum up (map-reduce) the deltas.
        if mp > 1:
            print('mp =', mp)
            pool = Pool(processes = mp)
        for epoch in range(num_epoch):
            if move:
                feat2id = {feat:pos for pos, feat in enumerate(selected_features)}
                n = len(feat2id)
                perceptron = Perceptron( n + 1 if bias else n )
            t = 0
            while t < train_set_ratio:
                if mp > 1:
                    inputs = []
                    labels = []
                    for _ in range(mp):
                        rand_inputs = [self._inputs[i] for i in train_set_idx[t:t + batch_size]]
                        rand_labels = [self._labels[i] for i in train_set_idx[t:t + batch_size]]
                        inputs.append(_vectorize(feat2id, rand_inputs, bias))
                        labels.append(np.asarray(rand_labels))
                        t += batch_size
                        if t >= train_set_ratio:
                            break
                    try:
                        tloss = perceptron.update_mp(inputs, labels, pool)
                    except Exception as e:
                        print(e)
                        continue
                else:
                    inputs = _vectorize(feat2id, train_x[t:t + batch_size], bias)
                    labels = np.asarray(train_y[t:t + batch_size])
                    t += batch_size
                    tloss = perceptron.update(inputs, labels)

                if valid_set_idx:
                    inputs = _vectorize(feat2id, validation_x, bias)
                    labels = np.asarray(validation_y)
                    vloss = np.average(perceptron.predict(inputs, labels))
                    if blend_loss is None:
                        blend_loss = vloss
                    else:
                        blend_loss = interpolate(vloss, blend_loss)
                print(f"Error rate epoch.{epoch}({t}/{train_set_ratio}|{total}): {tloss} {vloss}", flush = True)
                loss_info += f'{epoch},{tloss},{vloss}\n'
                shuffle(train_set_idx)
            # end while - batch

            if isinstance(blend_loss, float):
                if blend_loss < best_weights[0]:
                    best_weights = (blend_loss, perceptron.weights.copy(), feat2id.copy(), bias)
                    np.savez(self._name + f'.{n}', best_weights)
                if blend_loss < stop_loss:
                    print('reached the goal, blended: ', blend_loss)
                    break  # more selective

            proficiency_idx = np.argsort(perceptron.convergency)
            id2feats = {pos:feat for feat, pos in feat2id.items()}
            for tok, idx in feat2id.items():
                convergency_info += f'{epoch},"{tok}",{perceptron.weights[idx]},{perceptron.convergency[idx]}\n'
            if bias:
                convergency_info += f'{epoch},#BIAS{bias}#,{perceptron.weights[n]},{perceptron.convergency[n]}\n'
            selected_idx = np.where(np.logical_and(perceptron.convergency < 15, perceptron.weights != 0))[0]
            r = len(selected_features) - len(selected_idx)
            if r / n > decrease_threshold:
                move = True
                print(f"{r} features out of {n} removed." + ' ( %.2f%% )' % (100 * r / n))
                selected_features = {id2feats[i] for i in selected_idx if i != len(id2feats)}
            else:
                print("develop current perceptron")
                move = False
        # end for - epoch

        print("Train end. loss =", best_weights[0], len(best_weights[2]), 'features')
        return best_weights, dict(loss_info = loss_info, convergency_info = convergency_info)


    def test(self, fname, best_weights):
        inputs = []
        valid_loss, weights, feat2id, bias = best_weights
        perceptron = Perceptron(-1, weights)
        with open(fname) as fr:
            for line in fr:
                inputs.append(self._featurize(line))
        inputs = _vectorize(feat2id, inputs, 1)
        return perceptron.predict(inputs)

    def csv(self, fname):
        with open(fname, 'w') as fw:
            fw.write('tok,count\n')
            for tok, cnt in  self._features.items():
                fw.write(f'"{tok}",{cnt}\n')


    def __str__(self):
        s = 'A perceptron wrapper\n'
        s += '\tfeatures size: %d\n' % len(self._features)
        s += '\tCorpus size: %d\n' % len(self._inputs)
        return s

def _split(x):
    x = x.strip().split()
    x = tuple(map(stem, x))
    return Counter(x) + Counter(' '.join(ng) for ng in n_gram(2, x))


if __name__ == '__main__':
    w = Trainer("data_titles", _split)
    w.add_corpus('../../data/titles-en-train.labeled')
    #w.add_corpus('../../test/03-train-input.txt')
    print(w)
    w.csv('count.csv')
    best_weights, info = w.train(bias = 1)
    with open('loss.csv', 'w') as fw:
        fw.write(info['loss_info'])
    with open('convergency.csv', 'w') as fw:
        fw.write(info['convergency_info'])
    res = w.test('../../data/titles-en-test.word', best_weights)
    with open("my.labels", "w") as fw:
        for i in res:
            fw.write('%d\n' % i)
