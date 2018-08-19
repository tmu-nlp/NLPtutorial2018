from collections import defaultdict
from tqdm import tqdm
import dill
import pickle
import random
import sys


epoch = int(sys.argv[1])

def CreateFeatures(x,y):
    phi = defaultdict(lambda:0)
    for i in range(len(y)):
        if i == 0:
            first_tag = "<s>"
        else:
            first_tag = y[i-1]
        if i == len(y):
            next_tag = "</s>"
        else:
            next_tag = y[i]
        phi += CreateTrans(first_tag,next_tag)
    for i in range(len(y)-1):
        phi += CreateEmit(y[i],x[i])
    return phi



def CreateEmit(y,x):
    phi_emit = defaultdict(lambda: 0)
    phi_emit[(y, x)] += 1
    phi_emit[y] += 1

    return phi_emit



def CreateTrans(first_tag,next_tag):
    phi_trans = defaultdict(lambda: 0)
    phi_trans[(first_tag, next_tag)] += 1

    return phi_trans

def hmm_viterbi(weight,x,possible_tags,dict_transition):
    l = len(x)
    best_score = dict()
    best_edge = dict()
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None
    for i in range(l):
        for prev in possible_tags.keys():
            for next_tag in possible_tags.keys():
                transition = 'T {} {}'.format(prev, next_tag)
                emission = 'E {} {}'.format(next_tag, x[i])
                if '{} {}'.format(i,prev) in best_score and transition in dict_transition:
                    score  = best_score['{} {}'.format(i,prev)]
                    score += weight[transition] 
                    score += weight[emission] 
                    if '{} {}'.format(i+1,next_tag) not in best_score or best_score['{} {}'.format(i+1, next_tag)] < score:
                        best_score['{} {}'.format(i+1, next_tag)] = score
                        best_edge['{} {}'.format(i+1, next_tag)] = '{} {}'.format(i, prev)
     
    #文末処理                    
    for prev in possible_tags.keys():
        transition = 'T {} </s>'.format(prev)
        emission = 'E </s> </s>'
        if (str(l)+' '+prev) in best_score and transition in dict_transition:
            score = best_score[str(l)+' '+prev]+ weight[transition] + weight[emission]
            if (str(l+1)+' </s>') not in best_score or best_score[str(l+1)+' </s>'] < score:
                best_score[str(l+1)+' </s>'] = score
                best_edge[str(l+1)+' </s>'] = '{} {}'.format(l, prev)
    #後ろ向きステップ
    tags = list()
    next_edge = best_edge[str(l+1)+' </s>']
    while next_edge != '0 <s>':
        position = next_edge.split()[0]
        tag = next_edge.split()[1]
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags



def main():
    #w = defaultdict(lambda:0)
    weight = defaultdict(int)
    l = 5
    data = list()
    possible_tags = dict()
    dict_t_e = defaultdict(int)
    dict_transition = defaultdict(int)
    with open('../../data/wiki-en-train.norm_pos') as i_f:
        for line in i_f:
            temp_list = list()
            words_tags = line.strip().split()
            previous = '<s>'
            word_list = list()
            tag_list = list()
            for word_tag in words_tags:
                word, tag = word_tag.split('_')
                possible_tags[tag] = 1
                transe = 'T {} {}'.format(previous, tag)
                dict_transition[transe] = 1
                previous = tag
                word_list.append(word)
                tag_list.append(tag)
            dict_transition['T {} </s>'.format(tag)] = 1
            temp_list.append((word_list, tag_list))
            data.append(temp_list)
        possible_tags['<s>'] = 1
        possible_tags['/<s>'] = 1
    dict_transition = dict(dict_transition)

    for _ in tqdm(range(epoch)):
        for i, line in enumerate(data):
            if i % 100 == 0:
                print (i) 
            for x,y_prime in enumerate(data):
                y_hat = hmm_viterbi(weight,x,possible_tags,dict_transition)
                phi_prime = CreateFeatures(x,y_prime)
                phi_hat = CreateFeatures(x,y_hat)
                weight += phi_prime - phi_hat

            
    with open('weight_aniki.dump', 'wb') as o_f:
        pickle.dump((dict(weight),dict_transition,possible_tags), o_f)

if __name__=='__main__':
    main()
