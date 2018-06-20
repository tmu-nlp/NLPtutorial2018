import math
from collections import defaultdict

V = 10**6
lam1 = 0.95
lam2 = 0.95
H = 0
W = 0

path_test = '../../data/wiki-en-test.word'

# モデル読み込み
probs = defaultdict(lambda:0)
witten_lambda = defaultdict(lambda:0)
with open('model','r',encoding = 'utf-8') as model_file:
    for line in model_file:
        line = line.strip()
        model = line.split('\t')
        probs[model[0]] = float(model[1])
        witten_lambda[model[0]] = float(model[2])

# グリッドサーチ
entropy = {}
for lam1 in [0.01 * n for n in range(1, 100)]:
    for lam2 in [0.01 * n for n in range(1, 100)]:
        with open(path_test,'r',encoding = 'utf-8') as test_file:
            for line in test_file:
                line = line.strip()
                words = line.split(' ')
                words.insert(0,'<s>')
                words.append('</s>')
                    
                for i in range(1, len(words)):
                    bigram = words[i-1] + ' ' + words[i]
                    P1 = lam1 * probs[words[i]] + (1-lam1)/V 
                    P2 = lam2 * probs[bigram] + (1-lam2) * P1 
                    H += -math.log2(P2)
                    W += 1

            entropy[(lam1,lam2)] = H/W

min_en = min(entropy.values())
opt_lam = [key for key, value in entropy.items() if value == min_en][0]

print(f'optimaized entropy = {min(entropy.values())}')
print(f'lam1 = {opt_lam[0]}')
print(f'lam1 = {opt_lam[1]}')
    
# witten-bell
# with open(path_test,'r',encoding = 'utf-8') as test_file:
#     for line in test_file:
#         line = line.strip()
#         words = line.split(' ')
#         words.insert(0,'<s>')
#         words.append('</s>')
            
#         for i in range(1, len(words)):
#             bigram = words[i-1] + ' ' + words[i]
#             P1 = witten_lambda[words[i]] * probs[words[i]] + (1 - witten_lambda[words[i]])/V 
#             P2 = witten_lambda[bigram] * probs[bigram] + (1 - witten_lambda[bigram]) * P1 
#             H += -math.log2(P2)
#             W += 1
#     print(f'entropy = {H/W}')

        
# 11.2848 0.95 0.95

