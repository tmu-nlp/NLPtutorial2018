# -*- coding: utf-8 -*-

import sys
import math
import random
from collections import defaultdict

def sampleone(probs):
    z = sum(probs)
    #remaining = random.randint(z)
    remaining = random.random()*z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i

def initialize(file,num_topics):
    xcorpus = []
    ycorpus = []
    xcounts = defaultdict(int)
    ycounts = defaultdict(int)
    N = defaultdict(int)
    with open(file) as f:
        for line in f:
            docid = len(xcorpus)
            words = line.split()
            topics = []
            for word in words:
                #topic = rand(num_topics)
                topic = random.randint(0,num_topics)
                topics.append(topic)
                xcounts,ycounts = Addcounts(word, topic, docid, 1,xcounts,ycounts)
                N[word]
            xcorpus.append(words)
            ycorpus.append(topics)
    return xcorpus,ycorpus,xcounts,ycounts,len(N)


def Addcounts(word, topic, docid, amount, xcounts, ycounts):
    xcounts["topic"] += amount
    xcounts["word|topic"] += amount
    ycounts["docid"] += amount
    ycounts["topic|docid"] += amount
    return xcounts,ycounts

def sample():
    test_file = '../../test/07-train.txt'
    alpha = 0.01
    beta = 0.01
    epoch = 10
    num_topics = int(sys.argv[1])
    Ny = num_topics
    xcorpus,ycorpus,xcounts,ycounts,Nx = initialize(test_file, num_topics)
    for _ in range(epoch):
        ll = 0
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                xcounts,ycounts = Addcounts(x,y,i,-1,xcounts,ycounts)
                probs = []
                for k in range(num_topics):
                    pw = (xcounts[str(x) + '|' + str(k)]  + alpha) / (xcounts[k] + alpha * Nx)
                    pt = (ycounts[str(k) + '|' + str(i)] + beta) / (ycounts[i] + beta * Ny)
                    probs.append(pw * pt)
                new_y = sampleone(probs)
                ll += math.log(probs[new_y])
                xcounts,ycounts = Addcounts(x,new_y,i,1,xcounts,ycounts)
                ycorpus[i][j] = new_y
        print (ll)

if __name__ == '__main__':
    sample()
    


