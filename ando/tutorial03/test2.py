import math
from collections import defaultdict


words = defaultdict(lambda: 0)
wordcount =0
f = open("wiki-ja-train.word")
lines = f.readlines()
for line in lines:
    line = line.rstrip()
    line = line.split(" ")
    for i in line:
        words[i] += 1
        wordcount += 1
for i in words:
    words[i] = - math.log(words[i]/wordcount,2)


lam = 0.95
V = 1000000
for line in open("wiki-ja-test.txt"):
    best_edge = defaultdict()
    best_score = defaultdict()
    best_edge[0] = None
    best_score[0] = 0
    for end_num in range(1, len(line)):
        best_score[end_num] = 10 ** 10
        for begin_num in range(len(line)):
            if begin_num == end_num:
                break
            else:
                word = line[begin_num:end_num]
                if word in words or len(word) == 1:
                    probability = lam * words[word] + (1 - lam) / V
                    my_score = best_score[begin_num] + -math.log(probability)
                    if my_score < best_score[end_num]:
                        best_score[end_num] = my_score
                        best_edge[end_num] = [begin_num, end_num]

    words2 = []
    next_edge = best_edge[len(best_edge) - 1]
    while next_edge is not None:
        word = line[next_edge[0]:next_edge[1]]
        words2.append(word)
        next_edge = best_edge[next_edge[0]]
    words2.reverse()
    sentence = ' '.join(words2)
