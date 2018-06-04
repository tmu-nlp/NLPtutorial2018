from collections import defaultdict
import sys

def create_features(x):
	phi=defaultdict(int)
	words=x.split(' ')
	for word in words:
		phi["UNI:"+word]+=1
	return phi

def predict_one(w,phi):
	score=0
	for name,value in phi.items():
		if name in w:
			score+=value*w[name]
	if score>=0:
		return 1

	else:
		return -1

def update_weights(w,phi,y):
	for name,value in phi.items():
		w[name]+= int(value)*int(y)



if __name__=="__main__":
	w = defaultdict(int)
	
	with open('../../data/titles-en-train.labeled', 'r') as input_file:
		for line in input_file:
			spl=line.strip().split("\t")
			x=spl[1]
			y=spl[0]

	
		phi=create_features(x)
		y2=predict_one(w,phi)
		if y2 != y:
			update_weights(w,phi,y)

	with open('model','w') as model:
		for name, weight in sorted(w.items()):
			model.write(f'{name}\t{weight}\n')

	
		