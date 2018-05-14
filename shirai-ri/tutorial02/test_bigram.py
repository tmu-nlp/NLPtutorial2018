import codecs
import math
from collections import defaultdict

prob_dic = defaultdict(int)
with codecs.open('model_file.txt', 'r', 'utf-8') as prob_file:
    for line in prob_file:
        line = line.strip().split('\t')
        prob_dic[line[0]] = float(line[1])
                
        
V = 1000000
W = 0
H = 0
lambda_1 = 0.95
lambda_2 = 0.95
        
with codecs.open('../../data/wiki-en-test.word', 'r', 'utf-8') as test_file:
    for line in test_file:
        words = line.strip().split()
        words.insert(0, '<s>')
        words.append('</s>')
        for i in range(1, len(words)):
            P1 = lambda_1 * prob_dic[words[i]] + (1 - lambda_1) / V
            P2 = lambda_2 * prob_dic["{} {}".format(words[i - 1], words[i])] + (1 - lambda_2) * P1
            H += -math.log2(P2)
            W += 1

    print("Entropy = {}".format(H / W))
