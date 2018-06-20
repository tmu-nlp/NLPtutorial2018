# -*- coding: utf-8 -*-

from collections import defaultdict
import re
import pickle
import numpy as np
from Network import Output, Hidden, Input
import cProfile


def train(path, n):
    ids = defaultdict(lambda: len(ids))
    feat_lab = []
    for line in open(path, 'r'):
        ans = int(line.split('\t')[0])
        sentence = line.split('\t')[1].strip('\n').lower()
        phi = defaultdict(int)
        phi = create_features(sentence, n, phi, ids)
        feat_lab.append([phi, ans])

    feat_list = []
    for feat in feat_lab:
        feat_list.append([create_phi_list(feat[0], len(ids)), feat[1]])

    output_size = 1
    hidden_size = 16
    input_size = len(ids)
    out = Output()
    hid = Hidden(w=np.random.uniform(-0.1, 0.1, (hidden_size, output_size)),
                 b=np.random.uniform(-0.1, 0.1, (1, 1)))
    inp = Input(w=np.random.uniform(-0.1, 0.1, (hidden_size, input_size)),
                b=np.random.uniform(-0.1, 0.1, (1, hidden_size)))

    epoch = 10
    for e in range(epoch):
        for i, (phi, ans) in enumerate(feat_list):
            forward_nn(phi, out, hid, inp)
            backward_nn(ans, out, hid, inp)
            # print(i)
        print(e)
        pickle.dump(hid, open(f'hidden_model_{e}', 'wb'))
        pickle.dump(inp, open(f'input_model_{e}', 'wb'))


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
    # train('../../data/titles-en-train.labeled', 1)
    cProfile.run("train('../../data/titles-en-train.labeled', 1)")
