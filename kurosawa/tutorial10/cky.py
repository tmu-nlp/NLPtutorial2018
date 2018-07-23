from collections import defaultdict
import math
import sys

def read_grammar(grammar_file):
    with open(grammar_file) as grammar_data:
        nonterm = []
        preterm = defaultdict(list)
        for rule in grammar_data:
            lhs, rhs, prob = rule.strip().split('\t')
            prob = float(prob)
            rhs = rhs.split(' ')
            if len(rhs) == 1:
                preterm[rhs[0]].append((lhs, -math.log2(prob)))
            else:
                nonterm.append((lhs,rhs[0],rhs[1],-math.log2(prob)))
    return dict(preterm),nonterm

def connect_term(input_file):
    max_score = 10 ** 5
    with open(input_file) as input_data:
        for line_len,line in enumerate(input_data):
            print(line_len)
            best_score = defaultdict(lambda: max_score)
            best_edge = defaultdict(tuple)
            words = line.split()
            for i in range(len(words)):
                try:
                    for lhs,prob in preterm[words[i]]:
                        best_score['{}\t{}_{}'.format(lhs,i,i+1)] = prob
                except:
                    pass
            for j in range(2,len(words)+1):
                for i in reversed(range(j-1)):
                    for k in range(i+1,j):
                        for sym,lsym,rsym,prob in nonterm:
                            lsym_be = '{}\t{}_{}'.format(lsym,i,k)
                            rsym_af = '{}\t{}_{}'.format(rsym,k,j)
                            if best_score[lsym_be] < max_score and best_score[rsym_af] < max_score:
                                my_lp = best_score[lsym_be]+best_score[rsym_af]+prob
                                sym_i_j = '{}\t{}_{}'.format(sym,i,j)
                                if my_lp < best_score[sym_i_j]:
                                    best_score[sym_i_j] = my_lp
                                    best_edge[sym_i_j] = (lsym_be,rsym_af)
#                                    print(sym_i_j)
            yield dict(best_edge),len(words),words

def s_tree_print_subroutine(sym_now,mode=0):
#    sym_now_1,sym_now_2 = sym_now
    
    print(mode)
    for sym_now_now in sym_now:
        if sym_now_now in best_edge:
            sym,edge = sym_now_now.split('\t')
#            if mode == 1:
#                yield('({} {})'.format(sym,s_tree_print_subroutine(best_edge[sym_now_now],0)))
#            else:
            return('({} {})'.format(sym,s_tree_print_subroutine(best_edge[sym_now_now])))
        else:
            sym,edge = sym_now_now.split('\t')
            edge = edge.split('_')
#            if mode == 1:
#                yield('({} {})'.format(sym,words[int(edge[0])]))
#            else:
            return('({} {})'.format(sym,words[int(edge[0])]))

'''
    print(sym_now_)
    for sym_now in sym_now_:
        if sym_now in best_edge:
            sym,edge = sym_now.split('\t')
            edge = edge.split('_')
            return '({} {} {})'.format(sym,s_tree_print_subroutine(edge[0]),s_tree_print_subroutine(edge[1]))
        else:
            sym,edge = sym_now.split('\t')
            edge = edge.split('_')
            return '({} {})'.format(sym,words[edge[0]])
'''
def s_tree_print(best_score,best_edge,l,words):
    best_edge_1 = best_edge['S\t0_{}'.format(l)]
    print(best_edge_1)
#    print(s_tree_print_subroutine(best_edge_1))
#    for string in s_tree_print_subroutine(best_edge_1,1):
    for string in s_tree_print_subroutine(('S\t0_{}'.format(l),'FIN')):
        print(string)

def s_tree_print_2_sub(sym_now):
    if sym_now in best_edge:
        sym,edge = sym_now.split('\t')
        edge = best_edge[sym_now]
        return '({} {} {})'.format(sym,s_tree_print_2_sub(edge[0]),s_tree_print_2_sub(edge[1]))
    else:
        sym,edge = sym_now.split('\t')
        edge = edge.split('_')
        return '({} {})'.format(sym,words[int(edge[0])])

def s_tree_print_2(best_edge,l,words):
    f_edge = best_edge['S\t0_{}'.format(l)]
    sym = 'S'
    edge = f_edge
    return('({} {} {})\n'.format(sym,s_tree_print_2_sub(edge[0]),s_tree_print_2_sub(edge[1])))

if __name__ == '__main__':
    try:
        if sys.argv[1] == 'test':
            grammar_file = '../../test/08-grammar.txt'
            input_file = '../../test/08-input.txt'
            output_file = 'test.trees'
    except:
        grammar_file = '../../data/wiki-en-test.grammar'
        input_file = '../../data/wiki-en-short.tok'
        output_file = 'short.trees'
    preterm, nonterm = read_grammar(grammar_file)
    with open(output_file,'w') as output_data:
        for best_edge,l,words in connect_term(input_file):
#            print('best_edge:{}'.format(best_edge))
            try:
                output_data.write(s_tree_print_2(best_edge,l,words))
            except:
                output_data.write('文章として成り立っていません({})\n'.format(' '.join(words)))
