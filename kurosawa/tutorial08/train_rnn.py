#train_rnn.py [epoch] [hidden_size] [training_rate]
import numpy as np
from collections import defaultdict
import pickle
import sys
import random
import math

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def find_max(p):
    y = 0
    for i in range(len(p)):
        if p[i] > p[y]:
            y = i
    return y

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

def create_one_hot(w, ids):
    vec = np.zeros((len(ids),1))
    if w in ids:
        vec[ids[w]] = 1
    return vec

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

def forward_rnn(net,x):
    h = []#hidden states list
    p = []#predicted states list (after softmax)
    y = []#predicted one-hot vector list
    w_rx, w_rh, b_r, w_oh, b_o = net
    for t in range(len(x)):
#exist previous hidden state
        if t > 0:
            h.append(np.tanh(np.dot(w_rx, x[t]) + np.dot(w_rh, h[t-1]) + b_r))
#first time step (no previous hidden state)
        else:
            h.append(np.tanh(np.dot(w_rx, x[t]) + b_r))
        p.append(softmax(np.dot(w_oh, h[t]) + b_o))
        y.append(find_max(p[t]))
    return h,p,y

def gradient_rnn(net,x,h,p,y):
    hidden = int(sys.argv[2])
    w_rx, w_rh, b_r, w_oh, b_o = net
#initialize gradient weights
    dw_rx = np.zeros((hidden,len(ids_x)))
    dw_rh = np.zeros((hidden,hidden))
    db_r = np.zeros((hidden,1))
    dw_oh = np.zeros((len(ids_y),hidden))
    db_o = np.zeros((len(ids_y),1))
    err_r_ = np.zeros((len(b_r),1))
    for t in reversed(range(len(x))):#gradient is calculated from back
        err_o_ = y[t] - p[t]
#hidden to output
        dw_oh += np.outer(err_o_, h[t])
        db_o += err_o_
        err_r = np.dot(w_rh, err_r_) + np.dot(w_oh.T, err_o_)
        err_r_ = err_r * (1 - h[t] ** 2)
#input to hidden
        dw_rx += np.outer(err_r_, x[t])
        db_r += err_r_
#if previous hidden state (previous hidden to current hidden)
        if t != 0:
            dw_rh += np.outer(err_r_, h[t-1])
        dnet = [dw_rx, dw_rh, db_r, dw_oh, db_o]
    return dnet

def update_weights(net,dnet,lambda_):
    for i in range(len(net)):
        net[i] += lambda_ * dnet[i] 
    return net

def train(epoch, ids_x, ids_y, train_file):
    lambda_ = float(sys.argv[3])
    hidden = int(sys.argv[2])
#making onehot vector list
    with open(train_file) as train:
        featlab = make_featlab(train, ids_x, ids_y)
#initialized weight (0.001) & bias (0)
    w_rx = (np.random.rand(hidden,len(ids_x)) - 0.5) / 500
    w_rh = (np.random.rand(hidden, hidden) - 0.5) / 500
    b_r = np.zeros((hidden,1))
    w_oh = (np.random.rand(len(ids_y), hidden) - 0.5) / 500
    b_o = np.zeros((len(ids_y),1))
    net = [w_rx,w_rh,b_r,w_oh,b_o]
#training
    for i in range(epoch):
        print(i)
#sentence sequence is shuffled
        random.shuffle(featlab)
        for x, y in featlab:
#forward step
            h, p, y_predict = forward_rnn(net,x)
#calculate gradient
            dnet = gradient_rnn(net,x,h,p,y)
#update weights
            net = update_weights(net,dnet,lambda_)
    return net

if __name__ == '__main__':
    train_file = '../../data/wiki-en-train.norm_pos'
    epoch = int(sys.argv[1])
#making ids
    ids_x, ids_y = make_ids(train_file)
#training
    net = train(epoch, ids_x, ids_y, train_file)
#saving
    with open('weight_file.byte','wb') as w, open('ids_x_file.byte','wb') as ids_x_data, open('ids_y_file.byte','wb') as ids_y_data:
        pickle.dump(net,w)#saving network
        pickle.dump(dict(ids_x),ids_x_data)#saving source ids
        pickle.dump(dict(ids_y),ids_y_data)#saving target ids
