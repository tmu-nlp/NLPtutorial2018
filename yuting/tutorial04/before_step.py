import math
import sys
from collections import defaultdict


transition=defaultdict(lambda:0)
emission=defaultdict(lambda:0)
possible_tags=defaultdict(lambda:0)

#transition={}
#emission={}
#possible_tags={}


with open(sys.argv[1],'r')as model_file:
	for line in model_file:
		#split line into type, context, word, prob?
		#line=line.strip()
		t_e, context, word, prob = line.strip("\n").split(" ")
		possible_tags[context]=1
		if t_e=="T":
			transition[context+" "+word]=float(prob)
		else:
			emission[context+" "+word]=float(prob)


with open(sys.argv[2], 'r') as testfile:
	for line in testfile:
		line=line.strip()
		words=line.split(" ")
		l=len(words)
		
		lam=0.95
		lam_unk=1-lam
		N=1000000
	
		best_score={}
		best_edge={}

		best_score["0 <s>"]=0
		best_edge["0 <s>"]="NULL"
		for i in range(l):
			for prev in possible_tags.keys():
				for nex in possible_tags.keys():
					if str(i)+" "+prev in best_score and prev+" "+nex in transition:
						score=best_score[str(i)+" "+prev] -math.log(transition[prev+" "+nex],2) -math.log(lam * emission[nex+" "+words[i]] + lam_unk / N,2)			
						#if best_score[“i+1 next”] is new or < score?
						if str(i+1) +" "+ nex not in best_score or best_score[str(i+1) +" "+ nex] > score:
							best_score[str(i+1) +" "+nex]=score
							best_edge[str(i+1) +" "+nex]=str(i)+" "+prev
		for prev in possible_tags.keys():
			if str(l)+" "+prev in best_score and prev+" "+"</s>" in transition:
				score=best_score[str(l)+" "+prev] -math.log(transition[prev+" "+"</s>"],2)
				if str(l+1) +" "+ "</s>" not in best_score or best_score[str(l+1) +" "+ "</s>"] > score:
					best_score[str(l+1) +" "+ "</s>"] = score
					best_edge[str(l+1) +" "+ "</s>"] = str(l)+" "+prev





		tags=[]
		next_edge=best_edge[str(l+1) +" "+ "</s>"]
		while next_edge!="0 <s>":
			
			#split next_edge into position, tag?
			position,tag=next_edge.split(" ")
			#for position,tag in next_edge.split("/t"):
			tags.append(tag)
			next_edge=best_edge[next_edge]
		tags.reverse()
		print(' '.join(tags))


#Accuracy: 90.29% (4120/4563)

#Most common mistakes:
#NNS --> NN	61
#NNP --> NN	39
#JJ --> NN	35
#VBN --> NN	22
#JJ --> DT	21
#NN --> JJ	11
#NN --> DT	9
#VBN --> JJ	9
#VBP --> VB	9
#VB --> NN	8



