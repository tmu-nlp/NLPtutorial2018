# -*- coding: utf-8 -*-

from collections import defaultdict
import re
import pickle
from Network import Output, Hidden, Input


def test(train_path, teat_path, n):
    ids = defaultdict(lambda: len(ids))
    feat_lab = []
    for line in open(train_path, 'r'):
        sentence = line.split('\t')[1].strip('\n').lower()
        phi = defaultdict(int)
        create_features(sentence, n, phi, ids)
    ids_len = len(ids)

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


def create_phi_list(phi, ids_len):
    phi_list = [0] * ids_len
    for k, v in phi.items():
        if k < ids_len - 1:
            phi_list[k] = v
    return phi_list


def forward_nn(phi, out, hid, inp):
    inp.insert_phi(phi)
    phi = inp.forward_nn()
    hid.insert_phi(phi)
    phi = hid.forward_nn()
    out.insert(phi)


def backward_nn(ans, out, hid, inp):
    delta2 = out.back_nn(ans)
    delta = hid.back_nn(delta2)
    inp.update(delta, lam=0.03)
    hid.update(delta2, lam=0.03)


def create_features(sentence, n, phi, ids):
    if n == 0:
        return phi
    sentence = remove_symbol(sentence)
    words = sentence.split()
    for i in range(len(words) - n + 1):
        n_gram = ''
        for j in range(n):
            n_gram += words[i + j] + ' '
        phi[ids[n_gram]] += 1
    return create_features(sentence, n - 1, phi, ids)


def remove_symbol(sentence):
    symbol_list = r'[!-/:-@[-`{-~]'
    sentence = re.sub(symbol_list, '', sentence)
    return sentence


if __name__ == '__main__':
    test('../../data/titles-en-train.labeled', '../../data/titles-en-test.word', 1)

# epoch = 10
# Accuracy = 94.048884%
