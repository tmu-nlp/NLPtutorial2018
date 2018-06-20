from collections import defaultdict, Counter
from math import exp
from random import randint

def loadDataSet(fname = '../../data/titles-en-train.labeled'):
    data = []; labels = []
    vocab = defaultdict(lambda: len(vocab))
    with open(fname) as fr:
        for line in fr:
            lineArr = line.strip().split('\t')
            labels.append(float(lineArr[0]))
            data.append(Counter(vocab[w] for w in lineArr[1].split()))
    return data, labels, vocab

def selectJrand(i,m):
    j = i
    while (j == i):
        j = randint(0, m - 1)
    return j

def predict(alphas, labels, data, i, b):
    simsum = b
    one = data[i]
    for a, t, x in zip(alphas, labels, data):
        if a == 0:
            continue
        sim = multiply(x, one)
        simsum += a * t * sim
    return simsum

clip = lambda a, H, L: min(max(a, L), H)
zip_common = lambda one, eno: one.keys() & eno.keys()
zip_union  = lambda one, eno: one.keys() | eno.keys()

def multiply(lhs, rhs):
    return sum(lhs[i] * rhs[i] for i in zip_common(lhs, rhs))

def substract(lhs, rhs):
    return {i:(lhs[i] - rhs[i]) for i in zip_union(lhs, rhs)}

def distance(one, eno):
    sub = substract(one, eno)
    return multiply(sub, sub)

def smo_platt(data, labels, C, toler, maxIter, verbose = False):
    b = 0
    total_n = len(data)
    alphas = [0 for _ in range(total_n)]
    peacefull = 0
    while (peacefull < maxIter):
        alphaPairsChanged = 0
        for i in range(total_n):
            Ei = predict(alphas, labels, data, i, b) - labels[i]
            # t*(y-t)>o  t>0:y>t+o t<0:y<t-o
            # t*(y-t)<-o t>0:y<t-o t<0:y>t+o
            if (labels[i]*Ei > toler) and (alphas[i] > 0) or \
                    (labels[i]*Ei < -toler) and (alphas[i] < C) :

                j = selectJrand(i, total_n)
                if (labels[i] != labels[j]):
                    adv = alphas[j] - alphas[i]
                    L = max(0, adv)
                    H = min(C, C + adv)
                else:
                    pee = alphas[j] + alphas[i]
                    L = max(0, pee - C)
                    H = min(C, pee)
                if L==H:
                    #print("L==H")
                    continue

                eta = distance(data[i], data[j])
                if eta <= 0:
                    continue

                Ej = predict(alphas, labels, data, j, b) - labels[j]
                # ei==0: a-=t(y-t) t>0:a-=e t<0:a+=e
                delta = labels[j] * (Ei - Ej) / eta
                new_j = clip(delta + alphas[j], H, L)

                if (abs(new_j - alphas[j]) < 0.00001):
                    #print( "j not moving enough")
                    continue

                alpha_i = alphas[i]
                alpha_j = alphas[j]

                #update i by the same amount as j in the oppostie direction
                alphas[j] += delta
                alphas[i] -= labels[j] * labels[i] * delta

                mii = multiply(data[i], data[i])
                mij = multiply(data[i], data[j])
                mji = mij
                mjj = multiply(data[j], data[j])

                b1 = b - Ei
                b1 -= labels[i] * (alphas[i] - alpha_i) * mii
                b1 -= labels[j] * (alphas[j] - alpha_j) * mij
                b2 = b - Ej
                b2 -= labels[i] * (alphas[i] - alpha_j) * mji
                b2 -= labels[j] * (alphas[j] - alpha_j) * mjj

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphaPairsChanged += 1
                if verbose and False:
                    print("%d pairs changed at %d" % (i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            peacefull += 1
        else:
            peacefull = 0
        if verbose:
            print("Peacefull iteration: %d" % peacefull)
    return b, alphas

def weights(alphas, labels, data):
    wdic = defaultdict(int)
    for a, t, xdic in zip(alphas, labels, data):
        if a > 0:
            for f, w in xdic.items():
                wdic[f] += a*t*w
    return wdic

def predict_raw(src, dst, vocab, weights, b):
    with open(src, 'w') as fw, open(dst) as fr:
        for line in fr:
            line = (vocab[w] for w in line.strip().split())
            line = Counter(line)
            res = multiply(line, weights) + b
            fw.write('%d\n' % (1 if res > 0 else -1))

if __name__ == '__main__':
    x, y, v = loadDataSet('../../data/titles-en-train.labeled')
    #x, y, v = loadDataSet('short.labeled')
    print('Vocab size', len(v))
    b, alphas = smo_platt(x, y, 200, 0.0001, 100, True)
    wdic = weights(alphas, y, x)
    rv = {i:v for v, i in v.items()}
    for i, a in sorted(enumerate(alphas), key = lambda x:x[-1]):
        if a == 0:
            continue
        print(f"'{rv[i]}'\t{a}")
    predict_raw('svm.basic.labels', '../../data/titles-en-test.word', v, wdic, b)
