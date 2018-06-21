# -*- coding: utf-8 -*-

from collections import defaultdict
import pickle
from Network import Output
from train_nn import create_phi_list, forward_nn, remove_symbol


def test(teat_path, n):
    ids = pickle.load(open('ids.pkl', 'rb'))
    ids_len = len(ids)
    feat_lab = []
    for line in open(teat_path, 'r'):
        sentence = line.strip('\n').lower()
        phi = defaultdict(int)
        phi = create_features(sentence, n, phi, ids)
        feat_lab.append(phi)

    feat_list = []
    for feat in feat_lab:
        feat_list.append(create_phi_list(feat, ids_len))

    out = Output()
    hid = pickle.load(open('hidden_model_9', 'rb'))
    inp = pickle.load(open('input_model_9', 'rb'))

    with open('my_answer', 'w') as f:
        for i, phi in enumerate(feat_list):
            forward_nn(phi, out, hid, inp)
            result = out.result()
            print(result, file=f)


def create_features(sentence, n, phi, ids):
    if n == 0:
        return phi
    sentence = remove_symbol(sentence)
    words = sentence.split()
    for i in range(len(words) - n + 1):
        n_gram = ''
        for j in range(n):
            n_gram += words[i + j] + ' '
        if n_gram in ids:
            phi[ids[n_gram]] += 1
    return create_features(sentence, n - 1, phi, ids)


if __name__ == '__main__':
    test('../../data/titles-en-test.word', 1)

# epoch = 10
# Accuracy = 94.048884%
