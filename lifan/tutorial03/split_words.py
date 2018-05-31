# -*- coding: utf-8 -*-
import sys
import math
from collections import defaultdict
from train_model import train_unigram

train_file_path = sys.argv[1]
model_file_path = 'model.txt'
input_file_path = sys.argv[2]
output_file_path = 'output.txt'

lambda_1 = 0.95
V = 1000000

# train
train_unigram(train_file_path, model_file_path)

#modelの読み込み
probs = defaultdict(lambda: 0)
with open(model_file_path, 'r') as model_file:
	for line in model_file:
		line = line.strip()
		word_p = line.split("\t")
		probs[word_p[0]] = float(word_p[1])

# 出力ファイルの初期化
with open(output_file_path, 'w')as output_file:
	best_edge = defaultdict(lambda: 0)
	best_score = defaultdict(lambda: 0)
	with open(input_file_path, 'r') as input_file:
		for line in input_file:
			# 前向きステップ
			line = line.strip()
			best_edge[0] = None
			best_score[0] = 0
			for word_end in range(1,len(line)+1):
				best_score[word_end] = 10**10
				for word_begin in range(0, word_end):
					word = line[word_begin:word_end]
					if word in probs or len(word) == 1:
						prob = lambda_1 * probs[word] + (1 - lambda_1) / V
						my_score = best_score[word_begin] + (-math.log(prob,2))
						if my_score < best_score[word_end]:
							best_score[word_end] = my_score
							best_edge[word_end] = (word_begin, word_end)

			# 後ろ向きステップ
			words = []
			next_edge = best_edge[len(best_edge)-1]
			while next_edge != None:
				word = line[next_edge[0]:next_edge[1]]
				words.append(word)
				next_edge = best_edge[next_edge[0]]
			words.reverse()
			sentence = ' '.join(words)
			print(sentence, file=output_file)