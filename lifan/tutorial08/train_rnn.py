from collections import defaultdict
import math
import random
import numpy as np
import pickle

def CREATE_IDS(data_train):
    for line in data_train:
        words_poses = line.split()
        for word_pos in words_poses:
            word, pos = word_pos.split('_')
            # word = word.lower()
            x_ids['UNI:' + word]
            y_ids[pos]


def CREATE_FEAT_LAB(data_train):
    feat_lab = []
    for line in data_train:
        x, y = [], []
        words_poses = line.split()
        for word_pos in words_poses:
            word, pos = word_pos.split('_')
            # word = word.lower()
            x.append(CREATE_ONE_HOT(len(x_ids), x_ids['UNI:' + word]))
            y.append(y_ids[pos])
        feat_lab.append([x, y])

    return feat_lab


def CREATE_ONE_HOT(len_one_hot, id_one):
    one_hot = np.zeros(len_one_hot)
    one_hot[id_one] = 1

    return one_hot


def init_network(num_node, delta=0):
    np.random.seed(1)
    if delta == 0:
        w_rx = (np.random.rand(num_node, len(x_ids))-0.5)/5    # network[0][0]
        w_rh = (np.random.rand(num_node, num_node)-0.5)/5      # network[1][0]
        w_oh = (np.random.rand(len(y_ids), num_node)-0.5)/5    # network[2][0]
    else:
        w_rx = np.zeros((num_node, len(x_ids)))
        w_rh = np.zeros((num_node, num_node))
        w_oh = np.zeros((len(y_ids), num_node))
    b_r = np.zeros(num_node)                                   # network[0][1]
    b_o = np.zeros(len(y_ids))                                 # network[2][1]

    network = np.array([[w_rx, b_r], [w_rh], [w_oh, b_o]])

    return network


def softmax(scores):
    scores_softmax = []
    scores_exp = [math.exp(score) for score in scores]
    scores_exp_sum = sum(scores_exp)
    for score_exp in scores_exp:
        scores_softmax.append(score_exp / scores_exp_sum)

    return scores_softmax


def FORWARD_RNN(network, x):
    h, p, y_pre = [], [], []

    for t in range(len(x)):
        if t > 0:
            h.append(np.tanh(np.dot(network[0][0], x[t]) + np.dot(network[1][0], h[t-1]) + network[0][1]))
        else:
            h.append(np.tanh(np.dot(network[0][0], x[t]) + network[0][1]))
        p.append(softmax(np.dot(network[2][0], h[t]) + network[2][1]))
        y_pre.append(np.argmax(p[t]))

    return h, p, y_pre

def GRADIENT_RNN(network, x, y, h, p, num_node):

    delta_network = init_network(num_node, delta=1)
    gra_r = np.zeros(np.shape(network[0][1]))
    for t in range(len(x)-1, -1, -1):
        y_one_hot = CREATE_ONE_HOT(len(y_ids), y[t])
        gra_o = y_one_hot - p[t]
        delta_network[2][0] += np.outer(gra_o, h[t])
        delta_network[2][1] += gra_o
        error_r = np.dot(gra_r, network[1][0]) + np.dot(gra_o, network[2][0])
        gra_r = error_r*(1 - h[t]**2)
        delta_network[0][0] += np.outer(gra_r, x[t])
        np.outer(gra_r, x[t])
        delta_network[0][1] += gra_r
        if t != 0:
            delta_network[1][0] += np.outer(gra_r, h[t-1])
    return delta_network

def UPDATE_WEIGHTS(network, delta_network, rate_train):
    for i in range(len(network)):
        for j in range(len(network[i])):
            network[i][j] += rate_train*delta_network[i][j]

def train_nn(feat_lab, network, num_node):
    rate_train = 0.006
    for i, sentence in enumerate(feat_lab):
        x, y = sentence
        h, p, y_pre = FORWARD_RNN(network, x)
        delta_network = GRADIENT_RNN(network, x, y, h, p, num_node)
        UPDATE_WEIGHTS(network, delta_network, rate_train)

def train_nn_epoch(epoch, num_node, network):
    shuffled_feat_lab = list(feat_lab)
    random.seed(1)
    for num_epoch in range(epoch):
        random.shuffle(shuffled_feat_lab)
        train_nn(shuffled_feat_lab, network, num_node)

if __name__ == '__main__':
    path_data_train = '../../data/wiki-en-train.norm_pos'
    path_data_ids = 'train_rnn_ids.dump'
    path_data_network = 'train_rnn_network.dump'
    epoch = 20
    num_node = 8
    global x_ids, y_ids
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))

    with open(path_data_train, 'r') as data_train, open(path_data_ids, 'wb') as data_ids:
        CREATE_IDS(data_train)
        pickle.dump([dict(x_ids), dict(y_ids)], data_ids)

    with open(path_data_train, 'r') as data_train:
        global feat_lab
        feat_lab = CREATE_FEAT_LAB(data_train)

    network = init_network(num_node)

    train_nn_epoch(epoch, num_node, network)

    with open(path_data_network, 'wb') as data_network:
        pickle.dump(network, data_network)