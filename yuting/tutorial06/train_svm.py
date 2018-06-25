from collections import defaultdict
from tqdm import tqdm
from itertools import product

c=0.0001
margin=20
epoch=10

train_file='../../data/titles-en-train.labeled'
test_file='../../data/titles-en-test.word'

'''学習データの1行から素性を作成する'''
def create_feature(row_sentence):
	phi=defaultdict(lambda:0)
	words=row_sentence.split(' ')
	#1-gram
	for word in words:
		phi['UNI:'+word]+=1

	#2-gram
	for i in range(len(words)-1):
		phi['BI:'+words[i]+' '+words[i+1]]+=1
	return phi



#重みwを更新する
def update_weights(w,phi,y):
	for name,value in phi.items():
		w[name]+=value*y


#正則化は重みの使用時に行う#オンライン学習で L1 正則化
def getw(w,name,iter,c,last):
	if iter!=last[name]:
		c_size=c*(iter-last[name])
		if abs(w[name])<=c_size:
			w[name]=0
		else:
			w[name]-=(i if w[name] >= 0 else -1)*c_size
		last[name]=iter
	return w[name]


#マージンを用いたオンライン学習
def train_svm(train_file):
	w=defaultdict(lambda:0)
	#last=defaultdict(lambda:0)
	for _ in tqdm(range(epoch)):
		for i,line in enumerate(open(train_file,'r')):
			row_label,row_sentence=line.strip().split('\t')
			y=int(row_label)
			
			phi=create_feature(row_sentence)
			
			w=getw(w,name,iter,c,last)
			val=w*phi*y
			if val <= margin:
				update_weights(w,phi,y)
	with open('model','w')as f:
		f.writelines(f'{k}\t{v}\n' for k,v in sorted(w.items()))


def predict_one(w,phi):
	score=0
	for name,value in phi.items():
		if name in w:
			score+=value*w[name]
	if score>=0:
		return 1
	else:
		return -1

def test_svm(test_file):
	w=defaultdict(lambda:0)
	for line in open('model','r'):
		name,raw_value=line.strip().split('\t')
		w[name]=float(raw_value)
	with open('my_answer','w')as f:
		for line in open(test_file,'r'):
			row_sentence=line.strip()
			phi=create_feature(row_sentence)
			prediction=predict_one(w,phi)
			f.write(f'{prediction}\t{row_sentence}\n')


if __name__ == '__main__':
	train_svm(train_file)
	test_svm(test_file)
	

