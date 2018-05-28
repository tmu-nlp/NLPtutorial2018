from collections import defaultdict
import math

emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
context = defaultdict(lambda: 0)
transition2 = defaultdict(lambda: 0)
emission = defaultdict(lambda: 0)
possible_tags = defaultdict(lambda: 0)
for line in open("/Users/one/nlptutorial/data/wiki-en-train.norm_pos", 'r'):
    line = line.strip("\n")
    previous = "<s>"
    context["<s>"] += 1
    word_tags = line.split(" ")
    for word_tag in word_tags:
        word_tag = word_tag.split("_")
        word = word_tag[0].lower()
        transition[(previous, word_tag[1])] += 1
        context[word_tag[1]] += 1
        emit[(word, word_tag[1])] += 1
        previous = word_tag[1]
    context["</s>"] += 1
    transition[(previous, '</s>')] += 1

for key, value in transition.items():
    transition2[key] = value / context[key[1]]
for key, value in emit.items():
    emission[key] = value / context[key[1]]


for line in open('/Users/one/nlptutorial/data/wiki-en-test.norm', 'r'):
    lam = 0.95
    V = 1000000
    words = line.lower().strip('\n').split(" ")
    words.append('</s>')
    words.insert(0,'<s>')
    l = len(words)
    best_score = defaultdict(lambda: 10 ** 10)
    best_edge = defaultdict(str)
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None
    flag=0
    for i in range(0, l):
        for prev_tag in context.keys():
            for next_tag in context.keys():
                if flag == 0:
                    flag = 1
                    continue
                if best_score[(i, prev_tag)] != 10 ** 10 and transition2[(prev_tag, next_tag)] != 0:
                    score = best_score[(i, prev_tag)] + (-math.log(lam * transition2[(prev_tag, next_tag)] + ((1-lam) / V))) + (-math.log((lam * emission[(words[i], next_tag)]) + ((1-lam) / V)))
                    if best_score[(i+1 ,next_tag)] > score:
                        best_score[(i+1, next_tag)] = score
                        best_edge[(i+1, next_tag)] = (i, prev_tag)
                else:
                    score = best_score[(i, prev_tag)] + (-math.log(lam * transition2[(prev_tag, next_tag)] + ((1-lam) / V))) + (-math.log((lam * emission[(words[i], next_tag)]) + ((1-lam) / V)))
                    best_score[(i+1, next_tag)] = score
                    best_edge[(i+1, next_tag)] = (i, prev_tag)

            next_tag = '</s>'
            if best_score[(i, prev_tag)] != 10 ** 10 and transition2[(prev_tag, next_tag)] != 0:
                score = best_score[(i, prev_tag)] + (-math.log(lam * transition2[(prev_tag, next_tag)] + ((1-lam) / V))) + (-math.log((lam * emission[(words[i], next_tag)]) + ((1-lam) / V)))
                if best_score[(i+1 ,next_tag)] > score:
                    best_score[(i+1, next_tag)] = score
                    best_edge[(i+1, next_tag)] = (i, prev_tag)
            else:
                score = best_score[(i, prev_tag)] + (-math.log(lam * transition2[(prev_tag, next_tag)] + ((1-lam) / V))) + (-math.log((lam * emission[(words[i], next_tag)]) + ((1-lam) / V)))
                best_score[(i+1, next_tag)] = score
                best_edge[(i+1, next_tag)] = (i, prev_tag)

    tags = []
    next_edge = best_edge[(len(words),"</s>")]
    while next_edge != (0, "<s>"):
        if next_edge == (0, "<s>"):
            break
        tag = next_edge[1]
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    sentence = ' '.join(tags)