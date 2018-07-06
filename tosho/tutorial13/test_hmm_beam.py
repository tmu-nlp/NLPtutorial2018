'''
train:
python test_hmm_beam.py wiki.best.model 6 <../../data/wiki-en-test.norm >ans.pos
../../script/gradepos.pl ../../data/wiki-en-test.pos ans.pos
'''

from collections import defaultdict
from itertools import chain
from sklearn.externals import joblib
import sys, os

EPOCH = 20

def main():
    model_path = sys.argv[1]
    beam = int(sys.argv[2])

    w, transition, tags = joblib.load(model_path)

    for X in load_test_data(sys.stdin):
        Y = hmm_viterbi(w, X, transition, tags, beam)
        print(*Y)

def load_test_data(doc):
    for line in doc:
        yield line.strip().split()

def create_emit(tag, word):
    yield (f'E|{tag}|{word}', 1)

def create_trans(first_tag, next_tag):
    yield (f'T|{first_tag}|{next_tag}', 1)

def hmm_viterbi(w, X, transition, possible_tags, beam=2):
    best_score, best_edge = {}, {}
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None
    active_tags = [["<s>"]]

    for i, x in enumerate(chain(X, ['</s>'])):
        my_best = dict()
        for prev_tag in active_tags[-1]:
            for next_tag in (possible_tags if x != '</s>' else [x]):
                prev_node = f'{i} {prev_tag}'
                next_node = f'{i+1} {next_tag}'
                tag_trans = f'{prev_tag} {next_tag}'
                if prev_node in best_score and tag_trans in transition:
                    score = best_score[prev_node]
                    for k, v in create_trans(prev_tag, next_tag):
                        score += w[k] * v
                    for k, v in create_emit(next_tag, x):
                        score += w[k] * v
                    if next_node not in best_score or best_score[next_node] < score:
                        best_score[next_node] = score
                        best_edge[next_node] = prev_node
                        my_best[next_tag] = score
        *next_active_tags, = map(lambda a:a[0], sorted(my_best.items(), key=lambda a: a[1], reverse=True))
        next_beam = min(len(my_best), beam)
        active_tags.append(next_active_tags[:next_beam])
    
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