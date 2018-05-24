import math
import sys
from collections import defaultdict


#inputfile='./model_file.word'
#outputfile='segment_file'

#f=open(inputfile,'r')
#g=open(outputfile,'w')

with open(sys.argv[1],'r') as inputfile, open(sys.argv[2], 'r') as testfile:

	probabilities = defaultdict(int)

	for line in inputfile:
		line = line.split('\t')
		probabilities[line[0]] = float(line[1])

########################

	lam=0.95
	lam_unk=1-lam
	V=1000000
	for line in testfile:
		line=line.strip()
		#words=line.split(" ")
		#words-utf=unicode(words,"utf-8")
		#前向き
		best_edge={}
		best_score={}
		best_edge[0] = 'NULL'
		best_score[0] = 0 
		for word_end in range(1,len(line)):
			best_score[word_end]=10000000000
			for word_begin in range(0,word_end):
				word=line[word_begin:word_end]
				if word in probabilities or len(word)==1:
					prob=lam * probabilities[word] + lam_unk / V
					my_score=best_score[word_begin]-math.log(prob,2)
					if my_score < best_score[word_end]:
						best_score[word_end]=my_score
						best_edge[word_end]=(word_begin,word_end)
	    #後向き
		words=[]
		next_edge=best_edge[len(best_edge)-1]
		while next_edge!='NULL':
			word=line[next_edge[0]:next_edge[1]]
			word.encode(encoding='utf-8')
			words.append(word)
			next_edge=best_edge[next_edge[0]]
		words.reverse()
		print(' '.join(words))





