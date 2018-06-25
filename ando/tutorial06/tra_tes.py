
# coding: utf-8

# In[148]:


from collections import defaultdict
import numpy as np


# In[149]:


def ngram(text, n):
    return list(zip(*[text[i:] for i in range(n)]))


# In[150]:


def create_features(x):
    phi = defaultdict(int)
    words = x.split(" ")
    for word in words:
        phi[word] += 1
    bigram = ngram(words, 2)
    for i in bigram:
        phi[i] += 1
    return phi


# In[151]:


def predict_all(w, input_file="/Users/one/nlptutorial/data/titles-en-test.word"):
    with open("result.txt","w")as f:
        for x in open(input_file).readlines():
            x = x.strip()
            phi = create_features(x)
            y_d = predict_one(w, phi) 
            f.write(str(y_d)+"\t"+x+"\n")


# In[152]:


def predict_one(w, phi):
    score = 0
    for name, value in phi.items(): 
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1


# In[153]:


def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y


# In[154]:


def getw(w,name2,ite,last):
    c=0.0001
    if ite != last[name2]:
        c_size = c * (ite - last[name2])
        if abs(w[name2]) <= c_size:
            w[name2] = 0
        else:
            w[name2] -= np.sign(w[name2]) * c_size
        last[name2] = ite
    return w[name2]


# In[155]:


def gakushu(w,margin, ite, last, input_data="/Users/one/nlptutorial/data/titles-en-train.labeled"):
    for pair in open(input_data).readlines():
        pair = pair.split("\t")
        phi = create_features(pair[1].strip())
        val = 0
        for word in phi:
            val += getw(w,word,ite,last)*phi[word]
        val *= int(pair[0])
        if val <= margin:
            update_weights(w,phi,int(pair[0]))
    return w


# In[156]:


if __name__ == '__main__':
    weight = defaultdict(int)
    lastdict = defaultdict(int)
    m = 0
    for i in range(1,11):
        weight = gakushu(weight,m,i,lastdict)
    predict_all(weight)


# In[157]:


#93.978038%

