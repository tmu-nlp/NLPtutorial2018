import sys
import math
from collections import defaultdict

transition = defaultdict(int)
emission = defaultdict(float)
possible_tags = defaultdict(float)

lambda_1 = 0.95
V = 1000000

def sp(a,b):
	return f"{a} {b}"

with open(sys.argv[1], 'r') as model_file:
	for line in model_file:
		line = line.strip()
		splited_line = line.split(" ")
		wtype = splited_line[0]
		context = splited_line[1]
		word = splited_line[2]
		prob = splited_line[3]
		possible_tags[context] = 1
		if wtype == "T":
			transition[context + " " + word] = prob
		else:
			emission[context + " " + word] = prob

with open(sys.argv[2], 'r') as input_file, open("output.txt", 'w') as output_file:
	for line in input_file:
		line = line.strip()
		words = line.split(" ")
		l = len(words)
		best_score = defaultdict(float)
		best_edge = defaultdict(int)
		best_score["0 <s>"] = 0
		best_edge["0 <s>"] = None
		for i in range(0, l):
			for prev in possible_tags:
				for next_ in possible_tags:
					if sp(i,prev) in best_score and sp(prev,next_) in transition:
						prob_T = float(transition[sp(prev,next_)])
						prob_E = lambda_1 * float(emission[sp(next_, words[i])]) + (1 - lambda_1) / V
						score = best_score[sp(i,prev)] - math.log2(prob_T) - math.log2(prob_E)
						if sp(i+1, next_) not in best_score or (score <= best_score[sp(i+1, next_)]):
							best_score[sp(i+1, next_)] = score
							best_edge[sp(i+1, next_)] = sp(i, prev)
			for tag in possible_tags:
				if sp(l,tag) in best_score and sp(tag,"</s>") in transition:
					score = best_score[sp(l, tag)] - math.log2(float(transition[sp(tag, "</s>")]))
					
					if sp(l+1, "</s>") not in best_score or (score <= best_score[sp(l+1, "</s>")]):
						best_score[sp(l+1, "</s>")] = score
						best_edge[sp(l+1, "</s>")] = sp(l, tag)

		tags = []
		next_edge = best_edge[sp(l+1, "</s>")]

		while next_edge != "0 <s>":
			position = next_edge.split(" ")[0]
			tag = next_edge.split(" ")[1]
			tags.append(tag)
			next_edge = best_edge[next_edge]
		tags.reverse()

		print("".join(tags), file=output_file)


