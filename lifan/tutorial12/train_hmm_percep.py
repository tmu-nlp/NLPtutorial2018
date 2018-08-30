from collections import defaultdict
import pickle
import random

def create_features(x, y):
    phi = defaultdict(int)
    for i in range(len(y)+1):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = y[i-1]
        if i == len(y):
            next_tag = '</s>'
        else:
            next_tag = y[i]
        phi['T {} {}'.format(first_tag, next_tag)] += 1
    for i in range(len(y)):
        phi['E {} {}'.format(y[i], x[i])] += 1
    return phi

def create_trans(tag1, tag2):
    phi['T {} {}'.format(tag1, tag2)] += 1

def create_emit(y, x):
    phi['E {} {}'.format(y, x)] += 1

def hmm_viterbi(x):
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
     
    for prev in possible_tags.keys():
        transition = 'T {} </s>'.format(prev)
        emission = 'E </s> </s>'
        if (str(l)+' '+prev) in best_score and transition in dict_transition:
            score = best_score[str(l)+' '+prev]+ weight[transition] + weight[emission]
            if (str(l+1)+' </s>') not in best_score or best_score[str(l+1)+' </s>'] < score:
                best_score[str(l+1)+' </s>'] = score
                best_edge[str(l+1)+' </s>'] = '{} {}'.format(l, prev)
    tags = list()
    next_edge = best_edge[str(l+1)+' </s>']
    while next_edge != '0 <s>':
        position = next_edge.split()[0]
        tag = next_edge.split()[1]
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags

if __name__ == '__main__':
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
    for _ in range(l):
        print ('epoch {}'.format(_))
        random.shuffle(data)        
        for i, line in enumerate(data):
            if i % 100 == 0:
                print (i) 
            for x, y_prime in line:
                y_hat = hmm_viterbi(x)
                phi_prime = create_features(x, y_prime)
                phi_hat = create_features(x, y_hat)
                if i % 100 == 0:
                    with open('epoch_'+str(_), 'a') as xx:
                        xx.write(str(i)+'\n')
                        xx.write('正解\n')
                        xx.write(' '.join(y_prime)+'\n')
                        xx.write('予測\n')
                        xx.write(' '.join(y_hat)+'\n')
                        xx.write('重み\n')
                        for k,v in sorted(weight.items()):
                            xx.write('{} {}\n'.format(k,v))
                        xx.write('\n')
                for key, value in phi_prime.items():
                    weight[key] += value
                for key, value in phi_hat.items():
                    weight[key] -= value
    with open('weight_aniki.dump', 'wb') as o_f:
        pickle.dump((dict(weight),dict_transition,possible_tags), o_f)