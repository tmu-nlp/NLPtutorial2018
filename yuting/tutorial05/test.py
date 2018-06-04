from train import create_features
from train import predict_one
from collections import defaultdict
import sys


with open(sys.argv[1],'r')as modelfile,open(sys.argv[2],'r')as testfile:
	w=defaultdict(lambda:0)
	for line in modelfile:
		spl= line.strip().split('\t')
		w[spl[0]] =  int(spl[1])
	for x in testfile:
		phi=create_features(x)
		y2=predict_one(w,phi)
		print(y2)

