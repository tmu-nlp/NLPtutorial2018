# -*- coding: utf-8 -*-
from collections import defaultdict
import pickle
import numpy as np
from RNN import Output, Hidden, Input
from tqdm import tqdm


def train(path):
    feat_lab, output_size, input_size = create_data(path)

    out, hid, inp = create_layer(output_size, input_size)

    lam = 0.005
    epoch = 30
    for e in tqdm(range(epoch)):
        for x_list, y_list in feat_lab:
            forward(x_list, out, hid, inp)
            backward(y_list, out, hid, inp)
            update_weights(lam, hid, inp)
    pickle.dump(hid, open(f'hidden_model_{hid.w_h.shape[0]}_{e+1}_{lam}', 'wb'))
    pickle.dump(inp, open(f'input_model_{hid.w_h.shape[0]}_{e+1}_{lam}', 'wb'))


def create_data(path):
    x_ids, y_ids = make_ids(path)

    pickle.dump(dict(x_ids), open('x_ids.pkl', 'wb'))
    pickle.dump(dict(y_ids), open('y_ids.pkl', 'wb'))

    x_size = len(x_ids)
    y_size = len(y_ids)
    feat_lab = []
    for sentence in open(path, 'r'):
        x_list = []
        y_list = []
        for word in sentence.split():
            x, y = word.split('_')
            x_vec = create_one_hot(x.lower(), x_size, x_ids)
            y_vec = create_one_hot(y, y_size, y_ids)
            x_list.append(x_vec)
            y_list.append(y_vec)
        feat_lab.append([x_list, y_list])

    output_size = y_size
    input_size = x_size

    return feat_lab, output_size, input_size


def create_layer(output_size, input_size):
    hidden_size = 84
    out = Output()
    hid = Hidden(w_h=np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size)),
                 w_o=np.random.uniform(-0.1, 0.1, (hidden_size, output_size)),
                 b_o=np.random.uniform(-0.1, 0.1, (1, output_size)))
    inp = Input(w=np.random.uniform(-0.1, 0.1, (input_size, hidden_size)),
                b=np.random.uniform(-0.1, 0.1, (1, hidden_size)))
    return out, hid, inp


def initialize(out, hid, inp):
    out.initialize()
    hid.initialize()
    inp.initialize()


def forward(x_list, out, hid, inp):
    initialize(out, hid, inp)
    for t, x in enumerate(x_list):
        input_hidden = inp.forward(x)
        hidden_output = hid.forward(input_hidden, t)
        out.forward(hidden_output)


def backward(y_list, out, hid, inp):
    for i, y in enumerate(reversed(y_list), start=1):
        t = len(y_list) - i
        err_out = out.backward(y, t)
        err_hidden = hid.backward(err_out, t)
        inp.backward(err_hidden, t)


def update_weights(lam, hid, inp):
    hid.update(lam)
    inp.update(lam)


def make_ids(path):
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    for sentence in open(path, 'r'):
        for word in sentence.split():
            x, y = word.split('_')
            x_ids[x.lower()]
            y_ids[y]
    return x_ids, y_ids


def create_one_hot(feat, size, ids):
    vec = np.zeros(size)
    if feat in ids:
        vec[ids[feat]] = 1
    return vec


if __name__ == '__main__':
    # train('../../test/05-train-input.txt')
    train('../../data/wiki-en-train.norm_pos')
