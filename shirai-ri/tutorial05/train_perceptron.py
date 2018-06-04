
# coding: utf-8

# In[8]:

import sys
import codecs
from collections import defaultdict


def PREDICT_ONE(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w.keys():
            score += value * w[name]
    if score > 0:
        return 1
    else:
        return -1


def CREATE_FEATURES(x):
    phi = defaultdict(int)
    words = x.strip().split()
    for word in words:
        phi["UNI:" + word] += 1
    return phi


def UPDATE_WEIGHTS(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y

        
def TRAIN_PECEPTRON(train_file, model_file, epoch):
    w = defaultdict(int)
    for l in range(epoch):
        with codecs.open(train_file, 'r', 'utf8') as t_f:
            for line in t_f:
                y, x = line.strip().split('\t')
                y = int(y) # int型に直すのを忘れずに
                phi = CREATE_FEATURES(x)
                y_d = PREDICT_ONE(w, phi) # Yダッシュなので y_d
                if y_d != y:
                    UPDATE_WEIGHTS(w, phi, y)
    with codecs.open(model_file, 'w', 'utf8') as m_f:
        for key, value in w.items():
            m_f.write("{}\t{}\n".format(key, value))
    
    
if __name__ == '__main__':
    #TRAIN_PECEPTRON(sys.argv[1], sys.argv[2], sys.argv[3])
    TRAIN_PECEPTRON('./nlptutorial-master/data/titles-en-train.labeled', './model_file.txt', 10)


