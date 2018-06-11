# -*- coding: utf-8 -*-

from collections import defaultdict
import re
import math


def sigmoid(x):
    y = 1 / (1 + math.e**-x)
    return y


def sigmoid_derivative(x):
    y = (1-sigmoid(x)) * sigmoid(x)
    return y


def learn_weight(path, n):
    w = defaultdict(float)
    margin = 5
    c = 0.0001
    epoch = 30
    for i in range(epoch):
        last = defaultdict(lambda: 0)
        for i, line in enumerate(open(path, 'r')):
            phi = defaultdict(lambda: 0)
            ans = int(line.split('\t')[0])
            sentence = line.split('\t')[1].strip('\n').lower()
            phi = create_features(sentence, n, phi)
            val = get_val(w, phi, c, i, last) * ans
            if val <= margin:
                update_weights(w, phi, ans)
    return w


def get_val(w, phi, c, iter_, last):
    val = 0
    for n_gram, count in phi.items():
        val += getw(w, n_gram, c, iter_, last) * count
    return val


def create_features(sentence, n, phi):
    if n == 0:
        return phi
    sentence = remove_symbol(sentence)
    words = sentence.split()
    # words = remove_preposition(words)
    for i in range(len(words)-n+1):
        n_gram = ''
        for j in range(n):
            n_gram += words[i+j] + ' '
        phi[n_gram] += 1
    return create_features(sentence, n - 1, phi)


def update_weights(w, phi, ans):
    for n_gram, count in phi.items():
        if ans == 1:
            w[n_gram] += count * sigmoid_derivative(w[n_gram] * count)
        elif ans == -1:
            w[n_gram] -= count * sigmoid_derivative(w[n_gram] * count)


def getw(w, n_gram, c, iter_, last):
    if iter_ != last[n_gram]:
        c_size = c * (iter_ - last[n_gram])
        if abs(w[n_gram]) <= c_size:
            w[n_gram] = 0
        else:
            if w[n_gram] > 0:
                w[n_gram] -= sigmoid(w[n_gram]) * c_size
            else:
                w[n_gram] += sigmoid(w[n_gram]) * c_size
            # w[n_gram] -= sigmoid(w[n_gram]) * c_size
        last[n_gram] = iter_
    return w[n_gram]


def remove_symbol(sentence):
    symbol_list = r'[!-/:-@[-`{-~]'
    sentence = re.sub(symbol_list, '', sentence)
    return sentence


def remove_preposition(words):
    preposition_list = ['on', 'to', 'for', 'of']
    words = [word for word in words if word not in preposition_list]
    return words


def predict_one(w, phi):
    score = 0
    for n_gram, count in phi.items():
        if n_gram in w:
            score += count * w[n_gram]
    if score >= 0:
        return 1
    else:
        return -1


if __name__ == '__main__':
    # path = '../../test/03-train-input.txt'
    path = '../../data/titles-en-train.labeled'
    weight = learn_weight(path, 2)
    with open('model_file', 'w') as f:
        for w, value in sorted(weight.items(), key=lambda x: x[1]):
            print(f'{w}\t{value}', file=f)
