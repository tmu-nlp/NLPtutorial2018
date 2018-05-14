import sys
import math
from collections import defaultdict

l1=0.95
l2=0.95
V=1000000
W=0
H=0

f=open('model_file')
probs=f.read()
probs=defaultdict(lambda:0)

with open(sys.argv[1],'r') as test_file:
	for line in test_file:
		line=line.strip()
		words=line.split(' ')
		words.append("</s>")
		words.inser(0,"<s>")
		for i in range(1,len(words)):
			P1=l1 * probs[words[i]] + (i-l1)/V
			P2=l2 * probs[f'{words[i-1]}{words[i]}'] + (1-L2)*P1
			H += -math.log(P2)
			W += 1

print(f'entropy = {H/W}')
