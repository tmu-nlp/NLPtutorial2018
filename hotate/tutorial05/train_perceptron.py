# -*- coding: utf-8 -*-

from collections import defaultdict
import re


def learn_weight(path, n):
    w = defaultdict(int)
    loop = 40
    for i in range(loop):
        for line in open(path, 'r'):
            phi = defaultdict(lambda: 0)
            ans = int(line.split('\t')[0])
            sentence = line.split('\t')[1].strip('\n').lower()
            phi = create_features(sentence, n, phi)
            result = predict_one(w, phi)
            if result != ans:
                update_weights(w, phi, ans)
    return w


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


def predict_one(w, phi):
    score = 0
    for n_gram, count in phi.items():
        if n_gram in w:
            score += count * w[n_gram]
    if score >= 0:
        return 1
    else:
        return -1


def update_weights(w, phi, ans):
    for n_gram, count in phi.items():
        w[n_gram] += count * ans


def remove_symbol(sentence):
    symbol_list = r'[!-/:-@[-`{-~]'
    sentence = re.sub(symbol_list, '', sentence)
    return sentence


def remove_preposition(words):
    preposition_list = ['on', 'to', 'for', 'of']
    words = [word for word in words if word not in preposition_list]
    return words


if __name__ == '__main__':
    # path = '../../test/03-train-input.txt'
    path = '../../data/titles-en-train.labeled'
    weight = learn_weight(path, 2)
    with open('model_file', 'w') as f:
        for w, value in sorted(weight.items(), key=lambda x: x[1]):
            print(f'{w}\t{value}', file=f)
