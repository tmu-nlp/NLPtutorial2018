
# coding: utf-8

# In[4]:

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


def PREDICT_ALL(model_file, input_file):
    w = defaultdict(int)
    with codecs.open(model_file, 'r', 'utf8') as m_f, codecs.open(input_file, 'r', 'utf8') as i_f:
        for line in m_f:
            name, value = line.strip().split("\t")
            w[name] = int(value)
        for x in i_f:
            phi = CREATE_FEATURES(x)
            y = PREDICT_ONE(w, phi)
            # yield は return よりメモリ的にいいらしい
            # return は関数内の処理を終了して吐き出す
            # yield は関数内の処理を一時停止して吐き出す　らしい
            yield y

def TEST_PERCEPTRON(model_file, input_file, answer_file):
    with codecs.open(answer_file, "w", 'utf8') as a_f:
        for y in PREDICT_ALL(model_file, input_file):
            a_f.write("{}\n".format(y))


##################################################################3

if __name__ == '__main__':
    #PREDICT_ALL('./model_file.txt', './nlptutorial-master/data/titles-en-test.labeled', './my_answer.txt')
    TEST_PERCEPTRON(sys.argv[1], sys.argv[2], sys.argv[3])

# In[ ]:



