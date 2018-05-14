import sys
from collections import defaultdict
import codecs

counts = defaultdict(int)
context_counts = defaultdict(int)

with codecs.open('../../data/wiki-en-train.word', 'r', 'utf-8') as input_file:
    for line in input_file:
        words = line.strip().split()
        words.insert(0, '<s>')
        words.append('</s>')
        for i in range(1, len(words)):
            counts[words[i-1] + " " + words[i]] += 1
            context_counts[words[i-1]] += 1
            counts[words[i]] +=1
            context_counts[""] += 1
            
print(counts.items())

print(context_counts.items())

with open("model_file.txt", "w") as output_file:
    for ngram, count in sorted(counts.items(), key = lambda x: x[0]):
#         print(ngram)
#         print(count)
        words = ngram.split()
#         print(words)
#         print('')
        del words[-1]
        context = "".join(words)
#         print(context)
        print(ngram)
        print(counts[ngram])
        print(context)
        print(context_counts[context])
        probability = counts[ngram] / context_counts[context]
        output_file.write(ngram + "\t" + str(probability) + "\n")
        

