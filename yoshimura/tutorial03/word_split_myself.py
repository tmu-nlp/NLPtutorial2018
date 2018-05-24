import math
from collections import defaultdict

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
        best_edge = {} # node : nodeに来るbest_edge 
        best_score = {} # node : best_score

        line = line.rstrip()
        # 前向きステップ
        best_score[0] = 0
        best_edge[0] = ''
        for node in range(1, len(line) + 1):
            best_score[node] = 10**10
            # nodeにくるedgeを求める(edgeは単語列)
            edges = [line[i:node] for i in range(node)]
            for edge in edges:
                if edge in probs or len(edge) == 1:
                    prob = lam * probs[edge] + (1-lam)/V 
                    score = best_score[node - len(edge)] - math.log2(prob)
                    if score < best_score[node]:
                        best_score[node] = score
                        best_edge[node] = edge
        
        # 後ろ向きステップ
        best_path = []
        node = len(best_edge) - 1 # 最後のNode番号取得
        next_edge = best_edge[node] 
        while next_edge:
            best_path.append(next_edge) 
            node -= len(next_edge) # 次に見るノードを計算
            next_edge = best_edge[node]
        print(' '.join(best_path[::-1]))






                