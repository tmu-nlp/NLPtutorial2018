# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import pickle
import sys
import random
import math

epoch = int(sys.argv[1])
lam = float(sys.argv[3])
hidden = int(sys.argv[2])

def make_ids(train_file):
    with open(train_file) as f:
        ids_x = defaultdict(lambda: len(ids_x))
        ids_y = defaultdict(lambda: len(ids_y))
        for line in f:
            words = line.split()
            for word in words:
                x, y = word.split('_')
                x = x.lower()
                ids_x[x]
                ids_y[y]
    return ids_x, ids_y

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def find_max(p):
    y = 0
    for i in range(len(p)-1):
        if p[i] > p[y]:
            y = i
    return y

def create_one_hot(id,size):
    vec = np.zeros(size)
    vec[id] = 1
    return vec

def forward_rnn(net,x):
    w_rx, w_rh, b_r, w_oh, b_o = net
    h = []
    p = []
    y = []
    for t in range(len(x)-1):
        if t > 0:
            h[t] = np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t-1]) + b_r)
        else:
            h[t] = np.tanh(np.dot(w_oh,h[t]) + b_r)
        p[t] = np.tanh(np.dot(w_oh,h[t]) + b_o)
        y[t] = find_max(p[t])
    
    return h,p,y

def make_featlab(train, ids_x, ids_y):
    featlab = []
    for line in train:
        x_vec = []
        y_vec = []
        words = line.split()
        for word in words:
            x, y = word.split('_')
            x = x.lower()
            x_vec.append(create_one_hot(x,ids_x))
            y_vec.append(create_one_hot(y,ids_y))
        featlab.append((x_vec, y_vec))
    return featlab

def gradient_rnn(net,x,h,p,y):
    w_rx, w_rh, b_r, w_oh, b_o = net
    #initialize gradient weights
    dw_rx = np.zeros((hidden,len(ids_x)))
    dw_rh = np.zeros((hidden,hidden))
    db_r = np.zeros((hidden,1))
    dw_oh = np.zeros((len(ids_y),hidden))
    db_o = np.zeros((len(ids_y),1))
    err_r_ = np.zeros((len(b_r),1))
    for t in reversed(range(len(x))):
        err_o = y[t] - p[t]
        dw_oh += np.outer(h[t],err_o)
        db_o += err_o
        err_r = np.dot(err_r_,w_rh) + np.dot(err_o)
        err_r_ = err_r * (1 - h[t] ** 2)
        dw_rx += np.outer(x[t],err_r_)
        db_r += err_r_
        if t != 0:
            w_rh += np.outer(h[t - 1],err_r_)
        dnet = dw_rx,dw_rh,db_r,dw_oh,db_o
    return dnet

def update_weights(net,dnet,lam):
    for i in range(len(net)):
        net[i] += lam * dnet[i] 
    return net
    

def train(epoch, ids_x, ids_y, train_file):
    with open(train_file) as train:
        featlab = make_featlab(train, ids_x, ids_y)
    w_rx = (np.random.rand(hidden,len(ids_x)) - 0.5) / 500
    w_rh = (np.random.rand(hidden, hidden) - 0.5) / 500
    b_r = np.zeros((hidden,1))
    w_oh = (np.random.rand(len(ids_y), hidden) - 0.5) / 500
    b_o = np.zeros((len(ids_y),1))
    net = [w_rx,w_rh,b_r,w_oh,b_o]
    for i in range(epoch):
        for x, y in featlab:
            h, p, y_predict = forward_rnn(net,x)
            dnet = gradient_rnn(net,x,h,p,y)
            net = update_weights(net,dnet,lam)
    return net,ids_x,ids_y


if __name__ == '__main__':
    train_file = '../../data/wiki-en-train.norm_pos'
    
    ids_x, ids_y = make_ids(train_file)
    net = train(epoch, ids_x, ids_y, train_file)

    with open('weight_file.byte','w') as w, open('ids_x_file.byte','w') as ids_x_data, open('ids_y_file.byte','w') as ids_y_data:
        pickle.dump(net,w)
        pickle.dump(dict(ids_x),ids_x_data)
        pickle.dump(dict(ids_y),ids_y_data)


