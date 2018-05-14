import sys
import re
from collections import defaultdict

counts = defaultdict(lambda: 0)
context_counts = defaultdict(lambda: 0)

with open(sys.argv[1], 'r') as input_file:
	for line in input_file:
		line = line.strip().lower()
		line = re.sub(r' ,', '', line)
		words = line.split(' ')
		words.insert(0, "<s>")
		words.append("</s>")
		for i in range(1,len(words)):
			words_bi = "{} {}".format(words[i-1],words[i])
			counts[words_bi] += 1
			context_counts[words[i-1]] += 1
			counts[words[i]] += 1
			context_counts[""] += 1

with open(sys.argv[2], 'w') as model_file:
	for bigram, count in counts.items():
		bi_words_list = bigram.split(" ")
		if len(bi_words_list) > 1:
			context = bi_words_list[0]
		else:
			context = ""
		probability = count / context_counts[context]
		print("{}\t{}".format(bigram, probability), file = model_file)




# for key,value in counts.items():
# 	print(key, value)
# for key,value in context_counts.items():
# 	print(key, value)