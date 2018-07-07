'''
test:
python cky.py <../../test/08­input.txt
wiki:
python cky.py ../../data/wiki-en-test.grammar <../../data/wiki-en-short.tok >wiki-en-test.trees
'''

from collections import defaultdict
from math import log2
from sys import stdin, stderr, argv

def main():
    data = load_data()
    grammar = load_grammar() if len(argv) == 1 else load_grammar(argv[1])

    for line in data:
        s = predict_s_tree(line, grammar)
        print(s)

def predict_s_tree(line, grammar):
    nonterm, preterm = grammar

    best_score = {}
    best_edge = {}

    words = line.split()

    # preterm processing
    for i, word in enumerate(words):
        for lhs, proba in preterm[word]:
            best_score[f'{i}|{i+1}|{lhs}'] = proba

    # nonterm processing
    for j in range(2, len(words)+1):
        for i in range(j-2, -1, -1):
            for k in range(i+1, j):
                for sym, lsym, rsym, proba in nonterm:
                    key = f'{i}|{j}|{sym}'
                    l_key = f'{i}|{k}|{lsym}'
                    r_key = f'{k}|{j}|{rsym}'
                    if (l_key in best_score) and (r_key in best_score):
                        this_proba = best_score[l_key] + best_score[r_key] + proba
                        if (key not in best_score) or (this_proba > best_score[key]):
                            best_score[key] = this_proba
                            best_edge[key] = (l_key, r_key)

    return create_s_tree(f'0|{len(words)}|S', best_edge, words)

def create_s_tree(key, best_edge, words):
    sym = key.split('|')[2]

    if key in best_edge:
        lkey, rkey = best_edge[key]
        lstruct = create_s_tree(lkey, best_edge, words)
        rstruct = create_s_tree(rkey, best_edge, words)
        return f'({sym} {lstruct} {rstruct})'
    else:
        i = int(key.split('|')[0])
        return f'({sym} {words[i]})'

def load_data():
    data = [line.strip() for line in stdin]
    return data

def load_grammar(grammar_path='../../test/08-grammar.txt'):
    nonterm = []
    preterm = defaultdict(list)

    for line in open(grammar_path):
        # print(line)
        lhs, rhs, proba = line.split('\t')

        rhs = rhs.split(' ')
        proba = log2(float(proba))

        if len(rhs) == 1:
            # ['a'] = [('DT', log2(0.6)),...]
            preterm[rhs[0]].append((lhs, proba))
        else:
            # [(S, NP, VP, log2(0.6)),...]
            nonterm.append((lhs, rhs[0], rhs[1], proba))
    
    # give unk an average proba over all syms
    unk_proba = defaultdict(list)
    sum_proba = 0.
    for rhs, lhs_list in preterm.items():
        for lhs, proba in lhs_list:
            unk_proba[lhs].append(proba)
            sum_proba += proba
    for rhs, probas in unk_proba.items():
        unk_proba[rhs] = sum(probas) / sum_proba
    *unk_proba, = zip(unk_proba.keys(), unk_proba.values())
    
    preterm.default_factory = (lambda: unk_proba)

    return nonterm, preterm

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')