import sys
import math

from collections import defaultdict


counts=defaultdict(lambda:0)
contest_counts=defaultdict(lambda:0)

with open(sys.argv[1],'r') as train_file:
	for line in train_file:
		line=line.strip()
		words=line.split(" ")
		
		words.insert(0,"<s>")
		words.append("</s>")
		
		for i in range(1,len(words)):
			counts[f'{words[i-1]}{words[i]}']+=1
			contest_counts[f'{words[i-1]}']+=1
			counts[f'{words[i]}']+=1
			contest_counts[""]+=1

with open(sys.argv[2],'w') as model_file:
	for ngram,count in counts.items():
		words=ngram.split()
		del words[-1:]
		context=" ".join(words)
		probability=counts[ngram]/contest_counts[context]
		model_file.write(ngram + '\t'+str(probability)+'\n')