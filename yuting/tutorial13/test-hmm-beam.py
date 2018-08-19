# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import dill
import sys

def train(data_train):
    with open('../../data/wiki-en-train.norm_pos', 'r') as data_in:
        Prob_T, Prob_E, context = train(data_in)
    
    context, transition, emit, Prob_T, Prob_E = [defaultdict(int) for i in range(5)]
    for line in data_train:
        previous = '<s>'
        context[previous] += 1
        wordtags = line.split()
        for wordtag in wordtags:
            word, tag = wordtag.split('_')
            transition[previous + ' ' + tag] += 1
            context[tag] += 1
            emit[tag + ' ' + word] += 1
            previous = tag
        transition[previous + ' </s>'] += 1
    for key, value in transition.items():
        previous, word = key.split()
        Prob_T[key] = value/context[previous]
    for key, value in emit.items():
        tag, word = key.split()
        Prob_E[key] = value/context[tag]

    with open('result/result_train.dump', 'wb') as data_out:
        dill.dump([Prob_T, Prob_E, context], data_out)

    return Prob_T, Prob_E, context

def test(data_test, Prob_T, Prob_E, context, B_size):
    for line in data_test:
        # 前ステップ
        N = 1000000; rate_unk = 0.05;
        words = line.split()
        best_score = defaultdict(lambda: 10**10)
        best_score['0 <s>'] = 0
        best_edge = defaultdict(lambda: '')
        active_tags = [['<s>']]

        for i in range(0, len(words)):
            my_best = defaultdict(lambda: 0)
            for prev in active_tags[i]:
                for nex in context.keys():
                    score_key_prev = '{} {}'.format(i, prev)
                    prob_key = '{} {}'.format(prev, nex)
                    if score_key_prev in best_score.keys() and prob_key in Prob_T.keys():
                        P_E = rate_unk/N
                        if '{} {}'.format(nex, words[i]) in Prob_E.keys():
                            P_E += (1 - rate_unk)*Prob_E['{} {}'.format(nex, words[i])]
                        score = best_score[score_key_prev] + -math.log2(Prob_T[prob_key]) + -math.log2(P_E)
                        score_key_nex = '{} {}'.format(i+1, nex)
                        if score < best_score[score_key_nex]:
                            best_score[score_key_nex] = score
                            best_edge[score_key_nex] = score_key_prev
                            my_best[nex] = score
            active_tags.append(list(map(lambda y: y[0], sorted(my_best.items(), key=lambda x: x[1])[:B_size])))

        best_edge_end = ''
        for edge in best_edge.keys():
            if edge.split()[0] == str(len(words)):
                end_key = edge.split()[1] + ' </s>'
                if Prob_T[end_key] != 0:
                    score = best_score[edge] + -math.log2(Prob_T[end_key])
                else:
                    score = best_score[edge]
                if score < best_score['-1 </s>']:
                    best_score['-1 </s>'] = score
                    best_score[edge]
                    best_edge_end = edge
        best_edge['-1 </s>'] = best_edge_end
        # 後ろステップ
        tags = []
        next_edge = best_edge['-1 </s>']
        for i in range(len(best_edge)):
            if next_edge == '0 <s>':
                break
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        yield ' '.join(tags) + '\n'



if __name__ == '__main__':
    argv = sys.argv
    B_size = int(argv[1])
    with open('result/result_train.dump', 'rb') as data_in:
        Prob_T, Prob_E, context = dill.load(data_in)
    with open('../../data/wiki-en-test.norm', 'r') as data_test:
        with open('result/my_answer_{}.pos'.format(B_size), 'w') as data_out:
            for result_line in test(data_test, Prob_T, Prob_E, context, B_size):
                data_out.write(result_line)

    with open('../../data/wiki-en-train.norm_pos', 'r') as data_in:
        Prob_T, Prob_E, context = train(data_in)
    with open('result/result_train.dump', 'wb') as data_out:
        dill.dump([Prob_T, Prob_E, context], data_out)