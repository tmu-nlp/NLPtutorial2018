# -*- coding: utf-8 -*-

from collections import defaultdict
import math

def main():
    nonterm = []
    preterm = defaultdict(list)
    with open('../../data/wiki-en-test.grammar')as g:
        for rule in g:
            lhs,rhs,prob = rule.split('\t')
            rhs_symbols = rhs.split(" ")
            if len(rhs) == 1:
                preterm[rhs[0]].append((lhs, -math.log2(prob)))
            else:
                nonterm.append(lhs,rhs[0],rhs[1],math.log(prob))
        #return nonterm,preterm
    with open('../../data/wiki-en-short.tok')as f:
        for line in f:
            words = line.rstrip().split(' ')
            best_score = defaultdict(lambda: -math.inf)
            best_edge = {} 
            
            for i in range(len(words)):
                for lhs, log_prob in preterm[words[i]]:
                    best_score['{}\t{}_{}'.format(lhs,i,i+1)] = log_prob
                    for j in range(2, len(words)+1):  
                        for i in range(j-2, -1, -1):  
                            for k in range(i+1, j):  
                        
                                for sym, lsym, rsym, logprob in nonterm:
                                    
                                    if best_score['{}\t{}_{}'.format(lsym,i,k)] > -math.inf and best_score['{}\t{}_{}'.format(rsym,k,j)] > -math.inf:
                                        
                                        my_lp = best_score['{}\t{}_{}'.format(lsym,i,k)] + best_score['{}\t{}_{}'.format(rsym,k,j)] + logprob
                                        
                                        if my_lp > best_score['{}\t{}_{}'.format(sym,i,j)]:
                                            best_score['{}\t{}_{}'.format(sym,i,j)] = my_lp
                                            best_edge['{}\t{}_{}'.format(sym,i,j)] = ('{}\t{}_{}'.format(lsym,i,k),'{}\t{}_{}'.format(rsym,k,j))
            def s_exps(sym_i_j):
                sym, i, j = sym_i_j.split(' ')
                if sym_i_j in best_edge:  
                    return f'({sym} {s_exps(best_edge[sym_i_j][0])} {s_exps(best_edge[sym_i_j][1])})'
                else: 
                    return f'({sym} {words[int(i)]})'
                print(s_exps(f'S 0 {len(words)}'))

if __name__=='__main__':
    nonterm = []
    preterm = defaultdict(list)
    main()




