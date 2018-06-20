import numpy as np
from collections import defaultdict
import pickle
import sys

def make_ids(train_file):
    with open(train_file) as f:
        ids = defaultdict(lambda: len(ids))
        for line in f:
            y,x = line.split('\t')
            words = x.lower().split()
            for word in words:
                ids[word]
    return ids

def create_features(x,ids):
    phi = [0 for i in range(len(ids))]
    words = x.lower().split()
    for word in words:
        if word in ids:
            phi[ids[word]] += 1
    return phi

def forward_nn(net,phi0):
    phi = [0 for i in range(len(net)+1)]
    phi[0] = phi0
    for i in range(len(net)):
        w, b = net[i]
        phi[i+1] = np.tanh(np.dot(w,phi[i])+b)
#    print(phi[-1])
    return phi

def backward_nn(net,phi,y_):
    j = len(net)
    delta = [0 for i in range(j)]
    delta.append(np.array([y_-phi[j][0]]))
    delta_ = [0 for i in range(j+1)]
    for i in reversed(range(j)):
        delta_[i+1] = delta[i+1] * (1 - phi[i+1] ** 2)
        w,b = net[i]
        delta[i] = np.dot(delta_[i+1],w)
    return delta_

def update_weights(net,phi,delta,lambda_):
    for i in range(len(net)):
        net[i][0] += lambda_ * np.outer(delta[i+1],phi[i])
        net[i][1] += lambda_ * delta[i+1]
    return net

def train(train_file, ids):
    n = int(sys.argv[1])#epoch
    lambda_ = float(sys.argv[2])
    net = []
    w0 = (np.random.rand(2,len(ids)) - 0.5)/5
    b0 = np.zeros(2)
    w1 = (np.random.rand(1,2) - 0.5)/5
    b1 = np.zeros(1)
    net = [[w0,b0],[w1,b1]]
    for i in range(n):
        print("epoch:{}".format(i+1))
        num_err = 0
        num_all = 0
        with open(train_file) as f:
            for line in f:
                y,x = line.split('\t')
                y = int(y)
                phi0 = create_features(x,ids)
                phi = forward_nn(net,phi0)
                delta_ = backward_nn(net,phi,y)
                net = update_weights(net,phi,delta_,lambda_)
                num_all += 1
#                if y - phi[-1]
    return net

if __name__ == '__main__':
    train_file = '../../data/titles-en-train.labeled'
    ids = make_ids(train_file)
#ids作成済み
    net = train(train_file, ids)
    with open('weight_file.txt','wb') as w, open('id_file.txt','wb') as id_:
        pickle.dump(net,w)
        pickle.dump(dict(ids),id_)
