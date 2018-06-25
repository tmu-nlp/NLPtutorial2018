# python svm.py ../../data/titles-en-train.labeled ../../data/titles-en-test.word my_answer.txt

from collections import defaultdict
import sys

train_f = sys.argv[1]

def predict_all(w, input_f):
    with open(input_f, 'r') as i_f:
        for x in i_f:
            phi = dict()
            for key, value in create_features(x).items():
                phi[key.replace('UNI:','')] = value
            y_ = sign(predict_one(w, phi))
            yield ('{}\t{}'.format(y_,x))

#一事例に対する予測
def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value*w[name]
    return score

#sign作成
def sign(score):
    if score >= 0:
        return 1
    else:
        return -1

#素性作成
def create_features(x):
    phi = defaultdict(int)
    words = x.strip().split()
    for word in words:
        phi['UNI:'+word] += 1
    return phi

#重みの更新
def update_weights(w, phi, y, c, iteration, last):
    for name, value in phi.items():
        w[name] = getw(w, name, c, iteration, last)
        w[name] += int(value)*y

#重みの正則化
def getw(w, name, c, iteration, last):
    if iteration != last[name]:     #重みが古くなっていたら更新
        c_size = c*(iteration - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = iteration
    return w[name]


if __name__ == '__main__':

    input_f = sys.argv[2]
    output_f = sys.argv[3]

    w = defaultdict(int)
    last = defaultdict(int)
    l = 20 #iteration
    c = 10 ** -5
    margin = 1
    for i in range(l):
        print('いま' + str(i+1) + '回だよ！！！！！！！！！！')
        with open(train_f, 'r') as t_f:
            for line in t_f:
                y, x = line.strip().split('\t')
                phi = dict()
                for key, value in create_features(x).items():
                    phi[key.replace('UNI:', '')] = value
                val = predict_one(w, phi) * int(y)
                if val <= margin:
                    update_weights(w, phi, int(y), c, i, last)
    with open(output_f, 'w') as out_f:
        for text in predict_all(w, input_f):
            out_f.writelines(text)

