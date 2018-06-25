import numpy as np
from collections import defaultdict
import pickle
import sys
from tqdm import tqdm

#ids=defaultdict(lambda:len(ids))
train_file = '../../data/titles-en-train.labeled'
epoch=1
l=0.1

def make_ids(train_file):
	with open(train_file) as f:
		ids = defaultdict(lambda: len(ids))
		for line in f:
			y,x = line.split('\t')
			words = x.lower().split()
			for word in words:
				ids[word]
	return ids

def create_features(x):
	phi = [0 for i in range(len(ids))]
	words = x.lower().split()

	for word in words:
		if word in ids:
			phi[ids[word]] += 1
	return phi



#ニューラルネットの伝搬コード
def forward_nn(network,phi0):
	phi=[phi0]
	for i in range(len(network)-1):
		w,b=network[i]
		phi[i]=np.tanh(np.dot(w,phi[i-1])+b)
	return phi




#ニューラルネットの伝搬コード
def backward_nn(net,phi,y_):
	j = len(net)
	delta = [0 for i in range(j+1)]
	delta[-1] = np.array([y_ - phi[j][0]])
	delta_ = [0 for i in range(j+1)]
	for i in reversed(range(j)):
		delta_[i+1] = delta[i+1] * (1 - phi[i+1] ** 2)
		w,b = net[i]
		delta[i] = np.dot(delta_[i+1],w)
	return delta_


#重み更新のコード
def update_weights(net,phi,delta_,l):
	for i in range(len(net)-1):
		w,b=net[i]
		w+=l*np.outer(delta_[i+1],phi[i])
		b+=l*delta_[i+1]
	


def train(train_file):
	ids=make_ids(train_file)
	feat_lab=[]
	for i,line in enumerate(open(train_file,'r')):
		y,x=line.strip().split('\t')
		y=int(y)
		phi=create_features(x)
		feat_lab.append((phi,y))


	net = []
	w0 = (np.random.rand(2,len(ids)) - 0.5)/5
	b0 = np.zeros(2)
	w1 = (np.random.rand(1,2) - 0.5)/5
	b1 = np.zeros(1)
	net = [[w0,b0],[w1,b1]]
#学習を行う
	for _ in tqdm(range(epoch)):
		for phi0,y in feat_lab:
			phi=forward_nn(net,phi0)
			delta_=backward_nn(net,phi,y)
			net=update_weights(net,phi,delta_,l)

	with open('weight_file.txt','wb') as w, open('id_file.txt','wb') as id_:
		
		w.write(net)
		id_.write(ids)
	
if __name__=="__main__":
	train(train_file)