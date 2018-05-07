import sys
from collections import defaultdict

counts = defaultdict(lambda: 0)
total = 0

train_file = open(sys.argv[1], "r")

for line in train_file:
    words = line.strip().split()
    words.append("</s>")

    for w in words:
        counts[w] += 1
        total += 1

model_file = open("model-file", "w")

for word, count in sorted(counts.items()):
    probability = count / total
    print(f'{word} {probability}', file = model_file)

