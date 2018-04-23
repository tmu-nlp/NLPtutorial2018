import sys
import math

r_1 = 0.95
r_unk = 1 - r_1
V = 1000000
W = 0
H = 0
unk = 0

probabilities = {}

model_file = open(sys.argv[1], "r")
for line in model_file:
	line = line.strip()
	word_p = line.split("\t")
	probabilities[word_p[0]] = word_p[1]
model_file.close()

test_file = open(sys.argv[2], "r")
for line in test_file:
	line = line.strip()
	words = line.split(" ")
	words.append("</s>")
	for w in words:
		W += 1
		P = r_unk / V
		if w in probabilities:
			P += r_1 * float(probabilities[w])
		else:
			unk += 1
		H += -math.log(P,2)
test_file.close()

entropy = H / W
coverage = (W - unk) / W
print("entropy={}".format(entropy))
print("coverage={}".format(coverage))
	