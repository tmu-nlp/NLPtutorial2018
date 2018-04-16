#!/usr/bin/python3
import sys

word_counts = {}

my_file = open(sys.argv[1], "r")

for line in my_file:
	line = line.strip()
	if len(line) > 0:
		words = line.split(" ")
	else:
		words = ""

	for w in words:
		if w in word_counts:
			word_counts[w] = word_counts[w] + 1
		else:
			word_counts[w] = 1

for foo, bar in word_counts.items():
	print("%s \t %r" % (foo, bar))