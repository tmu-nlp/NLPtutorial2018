import sys
import math
from collections import defaultdict

p = defaultdict(lambda: 0)

model_file = open("model-file", "r")
for line in model_file:
    w = line.strip().split()
    p[w[0]] = float(w[1])

test_file = open(sys.argv[1], "r")
l1 = 0.95
l2 = 1 - l1
V = 1000000
W = 0
H = 0
nuk = 0
for line in test_file:
    words = line.strip().split()
    words.append("</s>")

    for w in words:
        W += 1
        P = l2 / V
        if w in p:
            P += l1 * p[w]
        else:
            nuk += 1
        H += -1 * math.log2(P)

entropy = H / W
coverage = (W - nuk) / W
print(f'entropy = {entropy}')
print(f'coverage = {coverage}')
