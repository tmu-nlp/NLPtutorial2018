from collections import defaultdict
import math

def ngram(text, n):
    return list(zip(*[text[i:] for i in range(n)]))

def create_features(x):
    phi = defaultdict(int)
    words = x.split(" ")
    for word in words:
        phi[word] += 1
    bigram = ngram(words, 2)
    for i in bigram:
        phi[i] += 1
    return phi

def predict_all(w, input_file="/Users/one/nlptutorial/data/titles-en-test.word"):
    with open("result.txt","w")as f:
        for x in open(input_file).readlines():
            x = x.strip()
            phi = create_features(x)
            y_d = predict_one(w, phi) 
            f.write(str(y_d)+"\t"+x+"\n")

def predict_one(w, phi):
    score = 0
    for name, value in phi.items(): 
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y

def gakushu(w, input_data="/Users/one/nlptutorial/data/titles-en-train.labeled"):
    for pair in open(input_data).readlines():
        pair = pair.split("\t")
        phi = create_features(pair[1].strip())
        y_d = predict_one(w,phi)
        if y_d != int(pair[0]):
            update_weights(w,phi,int(pair[0]))
    return w

if __name__ == '__main__':
    weight = defaultdict(int)
    for i in range(10):
        weight = gakushu(weight)
    predict_all(weight)

#93.871768%

