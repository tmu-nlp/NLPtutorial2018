
# coding: utf-8

# In[ ]:

import codecs
from collections import defaultdict
import math

def test_hmm(test_file, model_file, answer):
    
    transition = defaultdict(int)
    emission = defaultdict(int)
    possible_tags = defaultdict(int)
    
    lam = 0.95
    lam_unk = 1- lam
    V = 1000000
    
    
    with codecs.open(model_file, 'r', 'utf8') as model_f, codecs.open(test_file, 'r', 'utf8') as test_f, codecs.open(answer, 'w', 'utf8') as answer_f:
       
        # モデル読み込み
        for line in model_f:
            typ, context, word, prob = line.strip().split()
            possible_tags[context] = 1 # 可能なタグとして保存
            if typ == 'T':
                transition['{} {}'.format(context, word)] = float(prob)
            else:
                emission['{} {}'.format(context, word)] = float(prob)
                
        # 実際のテスト
        for line in test_f:
            words = line.strip().split()
            best_score = dict()
            best_edge = dict()
            best_score['0 <s>'] = 0
            best_edge['0 <s>'] = 'NULL'
            
            #前向き
            for i in range(0, len(words)):
                for prev in possible_tags.keys():
                    for nex in possible_tags.keys():
                        if '{} {}'.format(i, prev) in best_score and '{} {}'.format(prev,nex) in transition:
                            score = best_score['{} {}'.format(i, prev)] - math.log(transition['{} {}'.format(prev, nex)], 2) - math.log(lam * emission['{} {}'.format(nex, words[i])] + lam_unk/V, 2)
                            if '{} {}'.format(i+1, nex) not in best_score or best_score['{} {}'.format(i+1, nex)] > score:
                                best_score['{} {}'.format(i+1, nex)] = score
                                best_edge['{} {}'.format(i+1, nex)] = '{} {}'.format(i, prev)
            # 最後の処理
            for prev in possible_tags.keys():
                if '{} {}'.format(len(words), prev) in best_score and '{} </s>'.format(prev) in transition:
                    score = best_score['{} {}'.format(len(words), prev)] - math.log(transition['{} </s>'.format(prev)], 2)
                    if '{} </s>'.format(len(words) + 1) not in best_score or best_score['{} </s>'.format(len(words) + 1)] > score:
                        best_score['{} </s>'.format(len(words) + 1)] = score
                        best_edge['{} </s>'.format(len(words) + 1)] = '{} {}'.format(len(words), prev)
                        
            # 後ろ向き
            tags = []
            next_edge = best_edge['{} </s>'.format(len(words) + 1)]
            while next_edge != '0 <s>':
                position, tag = next_edge.split()
                tags.append(tag)
                next_edge = best_edge[next_edge]
            tags.reverse()
            answer_f.write(' '.join(tags) + '\n')
    
    
    
    
if __name__ == '__main__':
    test_hmm('./nlptutorial-master/data/wiki-en-test.norm', './model_file.txt', 'my_answer.pos')
    

