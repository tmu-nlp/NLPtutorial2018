import sys
from collections import defaultdict

context = defaultdict(int)
transition = defaultdict(float)
emit = defaultdict(float)

with open(sys.argv[1], 'r') as train_file:
	for line in train_file:
		line = line.strip()
		previous = "<s>"
		context[previous] += 1
		wordtags = line.split(" ")
		print(wordtags)
		for wordtag in wordtags:
			word = wordtag.split("_")[0]
			tag = wordtag.split("_")[1]
			transition[previous + " " + tag] += 1
			context[tag] += 1
			emit[tag + " " + word] += 1
			previous = tag
		transition[previous + " </s>"] += 1

with open("model.txt", 'w') as f:
	for key, value in transition.items():
		previous = key.split(" ")[0]
		word = key.split(" ")[1]
		f.write(f"T {key} {value/context[previous]} \n")
	for key, value in emit.items():
		previous = key.split(" ")[0]
		word = key.split(" ")[1]
		f.write(f"E {key} {value/context[previous]} \n")

