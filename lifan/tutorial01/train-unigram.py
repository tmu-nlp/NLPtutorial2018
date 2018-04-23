import sys
from collections import defaultdict

my_file = open(sys.argv[1], "r")
counts = defaultdict(lambda: 0)
total_count = 0
for line in my_file:
	line = line.strip()
	words = line.split(" ")
	words.append("</s>")
	for word in words:
		counts[word] += 1
		total_count += 1

model_file = open(sys.argv[2], "w")
for word, count in counts.items():
	probability = float(count) / total_count
	print('{}\t{}'.format(word, probability), file=model_file)

my_file.close()
model_file.close()