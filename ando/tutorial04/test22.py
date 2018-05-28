from collections import defaultdict


emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)
context = defaultdict(lambda: 0)
transition2 = defaultdict(lambda: 0)
emission = defaultdict(lambda: 0)
possible_tags = defaultdict(lambda: 0)
for line in open("/Users/one/nlptutorial/data/wiki-en-train.norm_pos", 'r'):
    line = line.strip("\n")
    previous = "<s>"
    context[previous] += 1
    word_tags = line.split(" ")
    for word_tag in word_tags:
        word_tag = word_tag.split("_")
        word = word_tag[0].lower()
        tag = word_tag[1]
        transition[previous + ' ' + word_tag[1]] += 1
        context[word_tag[1]] += 1
        emit[word_tag[1] + ' ' + word_tag[0]] += 1
        previous = word_tag[1]
    transition[previous + ' </s>'] += 1

