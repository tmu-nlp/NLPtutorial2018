import pickle
import train_hmm_percep

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
                if '{} {}'.format(i,prev) in best_score and transition in dict_t_e:
                    score  = best_score['{} {}'.format(i,prev)]
                    score += weight[transition] 
                    if emission in weight:
                        score += weight[emission] 
                    if '{} {}'.format(i+1,next_tag) not in best_score or best_score['{} {}'.format(i+1, next_tag)] < score:
                        best_score['{} {}'.format(i+1, next_tag)] = score
                        best_edge['{} {}'.format(i+1, next_tag)] = '{} {}'.format(i, prev)
     
    for prev in possible_tags.keys():
        transition = 'T {} </s>'.format(prev)
        if (str(l)+' '+prev) in best_score and transition in dict_t_e:
            score = best_score[str(l)+' '+prev]+ weight[transition] * dict_t_e[transition]
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
    with open('weights_aniki.dump', 'rb') as o_f:
        weight, dict_t_e, possible_tags = pickle.load(o_f)
        
    with open('../../data/wiki-en-test.norm') as i_f, open('my_anser_gati', 'w') as o_f:
        for line in i_f:
            words = line.strip().split()
            y_hat = hmm_viterbi(words)
            o_f.write(' '.join(y_hat)+'\n')
