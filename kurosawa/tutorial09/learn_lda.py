import sys
import math
import random
from collections import defaultdict

def initialize(file_name,num_topics):
    xcorpus = []
    ycorpus = []
    xcounts = defaultdict(int)
    ycounts = defaultdict(int)
    N = defaultdict(int)
    with open(file_name) as f:
        for line in f:
            docid = len(xcorpus)
            words = line.split()
            topics = []
            for word in words:
                topic = random.randint(0,num_topics)
                topics.append(topic)
                xcounts,ycounts = Addcounts(word, topic, docid, 1, xcounts, ycounts)
                N[word]
            xcorpus.append(words)
            ycorpus.append(topics)
    return xcorpus,ycorpus,xcounts,ycounts,len(N)

def Addcounts(word, topic, docid, amount, xcounts, ycounts):
    xcounts[topic] += amount
    xcounts[word + '|' + str(topic)] += amount
    ycounts[docid] += amount
    ycounts[str(topic) + '|' + str(docid)] += amount
    return xcounts,ycounts

def Sampleone(probs):
    z = sum(probs)
    remaining = random.random()*z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
        if i == (len(probs)-1):
            print('error:sampleone')
            exit()

if __name__ == '__main__':
#    test_file = '../../data/wiki-en-documents.word'
    test_file = '../../test/07-train.txt'
    epoch = int(sys.argv[2])
    num_topics = int(sys.argv[1])
    alpha = 0.01
    beta = 0.01
    Ny = num_topics
    xcorpus,ycorpus,xcounts,ycounts,Nx = initialize(test_file, num_topics)
    for e in range(epoch):
        print(e)
        ll = 0
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                xcounts,ycounts = Addcounts(x,y,i,-1,xcounts,ycounts)
                probs = []
                for k in range(num_topics):
                    pw = (xcounts[str(x) + '|' + str(k)]  + alpha) / (xcounts[k] + alpha * Nx)
#                    pt = (ycounts[str(k) + '|' + str(y)] + beta) / (ycounts[y] + beta * Ny)
                    pt = (ycounts[str(k) + '|' + str(i)] + beta) / (ycounts[i] + beta * Ny)
                    probs.append(pw * pt)
                new_y = Sampleone(probs)
                ll += math.log(probs[new_y])
                Addcounts(x,new_y, i, 1, xcounts,ycounts)
                ycorpus[i][j] = new_y
        print(ll)
    for i in range(len(xcorpus)):
        for j in range(len(xcorpus[i])):
            print('{}_{}\t'.format(xcorpus[i][j],ycorpus[i][j]),end='')
        print()
'''
    for i,j in sorted(xcounts.items()):
        if j != 0:
            print(i,j)
    for i,j in sorted(ycounts.items()):
        if j != 0:
            print(i,j)
'''
