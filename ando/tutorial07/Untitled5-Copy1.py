
# coding: utf-8

# In[559]:


import numpy as np
from collections import defaultdict


# In[560]:


def predict_one(net, phi0):
    phi = [0]*(len(net) + 1)
    phi[0] = phi0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    score = phi[len(net)][0]
    return 1 if score >= 0 else -1


# In[561]:


def update_weights(network, fai2, delta_d, lam2):
    for i in range(0,len(network)-1,-1):
        w, b = network[i]
        w += lam2 * np.outer( delta_d[i+1], fai2[i] )
        b += lam2 * delta_d[i+1]


# In[562]:


def create_features(x, ids):
    phi = [0]*len(ids)
    for word in x.split(" "):
        phi[ids[word]] += 1
    return phi


# In[563]:


def create_features_test(x, ids):
    phi = [0]*len(ids)
    for word in x.split(" "):
        phi[ids[word]] += 1
    return phi


# In[564]:


def forward_nn(network, fai0):
    fai2 = np.array([fai0, 0, 0])
    for i in range(0,len(network)):
        w, b = network[i]
        fai2[i+1] = np.tanh( np.dot( w, fai2[i] ) + b ).T
    return fai2


# In[565]:


def backward_nn(network, fai2, y):
    J = len(network)
    delta2 = np.zeros(J+1, dtype=np.ndarray)
    delta2=np.append(delta2, np.array([y-fai2[J][0]]))
    delta2_d = np.zeros(J+1, dtype=np.ndarray)
    for i in range(J,0,-1):
        delta2_d[i] = delta2[i]*(1-np.square(fai2[i])).T
        w, b = network[i-1]
        delta2[i-1] = np.dot(delta2_d[i], w)
    return delta2_d


# In[ ]:


if __name__ == '__main__':
    ids = defaultdict(lambda: len(ids))
    feat_lab = []
    lam=0.1
    for line in open("/Users/one/nlptutorial/data/titles-en-train.labeled").readlines():
        line = line.split("\t")
        for word in line[1].split():
            ids[word]
    for line in open("/Users/one/nlptutorial/data/titles-en-train.labeled").readlines():
        line = line.split("\t")
        feat_lab.append((create_features(line[1].strip(), ids), line[0]))
    net = [(np.random.rand(2, len(ids))*2-1,np.full(2, -1))]
    net.append((np.random.rand(1, 2)*2-1,np.full(1, -1)))
    for pair in feat_lab:
        fai = forward_nn(net, pair[0])
        delta = backward_nn(net, fai, int(pair[1]))
        update_weights(net, fai, delta, lam)
    feat_lab=[]
    for line in open("/Users/one/nlptutorial/data/titles-en-test.word").readlines():
        line = line.split("\t")
        feat_lab.append((create_features_test(line[1].strip(), ids), line[0]))
    with open("test","w")as f:
        for pair in feat_lab:
            f.write(predict_one(net,pair[1])+"\t"+pair[1]+"\n")

