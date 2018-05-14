import sys
import math

lam1 = 0.95
lam2 = 0.8
V = 10*6
W = 0
H = 0


# モデル読み込み
probs = {}
with open('model','r',encoding = 'utf-8') as model_file:
    for line in model_file:
        line = line.strip()
        model = line.split('\t')
        probs[model[0]] = float(model[1])

# 評価
with open(sys.argv[1],'r',encoding = 'utf-8') as test_file:
    for line in test_file:
        line = line.strip()
        words = line.split(' ')
        words.insert(0,'<s>')
        words.append('</s>')
        for i in range(1, len(words)):
            P1 = 1-lam1/V
            if words[i] in probs:
                P1 += lam1*probs[f'{words[i]}']
            P2 = (1-lam2)*P1
            if f'{words[i-1]} {words[i]}' in probs:
                P2 += lam2*probs[f'{words[i-1]} {words[i]}']
            H += -math.log2(P2)
            W += 1
    print(f'entropy = {H/W}')

