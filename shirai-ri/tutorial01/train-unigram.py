
# coding: utf-8

# In[10]:

import sys
from collections import defaultdict


counts = defaultdict(int)
total_count = 0

with open(sys.argv[1], 'r') as text:
    for line in text:
        words = line.split()
        words.append("</s>")
        for word in words:
            counts[word] += 1
            total_count += 1

with open("model_file.word", "w") as text:
    for word, count in counts.items():
        #.keys()はキー
        #.values()はバリュー
        probability = float(counts[word]/total_count)
        text.write(word + "\t" + str(probability) + "\n")


