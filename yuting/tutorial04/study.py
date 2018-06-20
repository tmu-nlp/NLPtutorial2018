import sys
import math
from collections import defaultdict

emit=defaultdict(lambda:0)
transition=defaultdict(lambda:0)
context=defaultdict(lambda:0)

with open(sys.argv[1],'r')as train_file:
	for line in train_file:
		line=line.strip()
		previous="<s>"
		context[previous]+=1
		wordtags=line.split(" ")
		for wordtag in wordtags:
			#split wordtag into word, tag with “_”
			word,tag = wordtag.split("_")
			transition[previous+" "+tag]+=1
			context[tag]+=1
			emit[tag+" "+word]+=1
			previous=tag
		transition[previous + " " + "</s>"]+=1

with open(sys.argv[2],'w') as model_file:
	for key,value in transition.items():
		pervious,tag = key.split(" ")
		model_file.write("T" + " " + str(key) + " " + str(value/context[previous]) + "\n")
	for key,value in emit.items():
		tag, word = key.split(" ")
		model_file.write("E" + " " + str(key) + " " + str(value/context[tag]) + "\n")

#python study.py ../../data/wiki-en-train.norm_pos model_file.txt
#less model_file.txt 


