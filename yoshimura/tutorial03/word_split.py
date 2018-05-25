import math
best_edge = {}
best_score = {}

path_model = '../tutorial01/model'
path_test = '../../test/04-input.txt'

# モデル読み込み
probs = {}
with open(path_model,"r") as model_file:
    for line in model_file:
        line = line.strip()
        model = line.split("\t")
        probs[model[0]] = float(model[1])

with open(path_test,'r') as data_file:
    for line in data_file:
        # 前向きステップ
        best_edge[0] = None
        best_score[0] = 0
        for word_end in range(1, len(line) + 1): 
            best_score[word_end] = 10**10
            for word_begin in range(word_end - 1):
                word = line[word_begin : word_end] # 部分文字列を取得
                if word in probs or len(word) == 1:
                    prob = probs[word]
                    score = best_score[word_begin] - math.log(prob)
                    if score < best_score[word_end]:
                        best_score[word_end] = score
                        best_edge[word_end] = (word_begin, word_end)
        # 後ろ向きステップ
        # words = []
        # next_edge = best_edge[len(best_edge) - 1]
        # while next_edge != None:
        #     word = line[next_edge[0] : next_edge[1]]
        #     words.append(word)
        #     next_edge = best_edge[next_edge[0]]
        # print(''.join(words[::-1]))

                