import math
from collections import defaultdict
best_edge = {}
best_score = {}

path_model = '../tutorial01/model'
path_test = '../../data/wiki-ja-test.txt'
# path_model = '../tutorial01/model_test' # テスト用モデルファイルパス
# path_test = '../../test/04-input.txt' # テスト用ファイルパス

lam = 0.98
V = 10**6

# モデル読み込み
probs = defaultdict(lambda:0)
with open(path_model,"r") as model_file:
    for line in model_file:
        line = line.rstrip()
        model = line.split("\t")
        probs[model[0]] = float(model[1])

with open(path_test,'r') as data_file:
    for line in data_file:
        line = line.rstrip()
        # 前向きステップ
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1, len(line) + 1): 
            best_score[word_end] = 10**10
            for word_begin in range(0, word_end):
                word = line[word_begin : word_end] # 部分文字列を取得
                if word in probs or len(word) == 1:
                    prob = lam * probs[word] + (1-lam)/V
                    score = best_score[word_begin] - math.log2(prob)
                    if score < best_score[word_end]:
                        best_score[word_end] = score
                        best_edge[word_end] = (word_begin, word_end)
        
        # print(best_score)
        # 後ろ向きステップ
        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge:
            word = line[next_edge[0] : next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        print(' '.join(words[::-1]))

                