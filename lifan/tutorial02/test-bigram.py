import sys
import math
import re
from collections import defaultdict

r_1 = 0.95
r_2 = 0.95
V = 1000000
W = 0
H = 0

probs = defaultdict(lambda: 0)

with open(sys.argv[1], 'r') as model_file:
	for line in model_file:
		line = line.strip()
		line_list = line.split("\t")
		probs[line_list[0]] = float(line_list[1])

with open(sys.argv[2], 'r') as test_file:
	for line in test_file:
		line = line.strip().lower()
		line = re.sub(r' ,', '', line)
		words = line.split(" ")
		words.insert(0, "<s>")
		words.append("</s>")
		for i in range(1,len(words)):
			P1 = r_1 * probs[words[i]] + (1 - r_1) / V
			P2 = r_1 * probs["{} {}".format(words[i-1], words[i])] + (1 - r_2) * P1
			H += -math.log(P2, 2)
			W += 1

print("entropy = {}".format( H / W ))