'''
train:
python test_hmm_a_star.py wiki.best.model <../../data/wiki-en-test.norm >ans.a_star.pos
../../script/gradepos.pl ../../data/wiki-en-test.pos ans.a_star.pos
'''

from itertools import chain
from sklearn.externals import joblib
import sys, os
from heapq import *

EPOCH = 20

def main():
    model_path = sys.argv[1]

    w, transition, tags = joblib.load(model_path)

    for X in load_test_data(sys.stdin):
        Y = hmm_viterbi(w, X, transition, tags)
        print(*Y)

def load_test_data(doc):
    for line in doc:
        yield line.strip().split()

def create_emit(tag, word):
    yield (f'E|{tag}|{word}', 1)

def create_trans(first_tag, next_tag):
    yield (f'T|{first_tag}|{next_tag}', 1)

def hmm_viterbi(w, X, transition, possible_tags):
    best_score, best_edge = {}, {}
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None

    h = []
    heappush(h, (0, '0 <s>'))
    
    # 全可能性を探索する
    # 素性を利用する方法の場合、ヒューリスティック関数は、
    # 各タイムステップにおける、高々のスコアを計算すればよい。
    while len(h) != 0:
        _, prev_node = heappop(h)
        i, prev_tag = prev_node.split(' ')
        i = int(i)
        x = '</s>' if i == len(X) else X[i] 

        # 今回更新するヒープのアイテム
        my_best_score = {}

        for next_tag in (possible_tags if x != '</s>' else [x]):
            next_node = f'{i+1} {next_tag}'
            tag_trans = f'{prev_tag} {next_tag}'
            if tag_trans in transition:
                score = best_score[prev_node]
                for k, v in create_trans(prev_tag, next_tag):
                    score += w[k] * v
                for k, v in create_emit(next_tag, x):
                    score += w[k] * v
                if next_node not in best_score or best_score[next_node] < score:
                    best_score[next_node] = score
                    best_edge[next_node] = prev_node
                    my_best_score[next_node] = score
        
        # 最後まで達した場合は追加しない（それ以上探索しないので）
        if x != '</s>':
            for node, score in my_best_score.items():
                heappush(h, (-score, node))

    next_edge = f'{len(X)+1} </s>'
    tags = []
    while next_edge != None:
        idx, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]
    
    tags.reverse()
    
    return tags[1:-1]    

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')