'''
train:
python train-hmm-percep.py test.pkl <../../test/05-train-input.txt
'''

from collections import defaultdict
from itertools import chain
from sklearn.externals import joblib
import sys, os

EPOCH = 20

def main():
    model_path = sys.argv[1]
    path_body, path_ext = os.path.splitext(model_path)

    w = defaultdict(int)
    data, transition, tags = load_data()

    data_size = len(data)
    disp_interval = max(1, data_size // 10)    

    for l in range(EPOCH):
        N, S = 0, 0
        for i, (X, Y_prime) in enumerate(data):
            Y_hat = hmm_viterbi(w, X, transition, tags)
            phi_prime = create_features(X, Y_prime)
            phi_hat = create_features(X, Y_hat)

            correct = all(map(lambda i: i[0]==i[1], zip(Y_hat, Y_prime)))

            if i % disp_interval == 0:
                print_debug(f'{i} | predict: {Y_hat} | actual:  {Y_prime}{"" if correct else "*"}')

            if not correct:
                update_dict(w, phi_prime.items())
                update_dict(w, phi_hat.items(), -1)

            N += 1
            if correct:
                S += 1
        acc = S / N
        print_debug(f'epoch {l+1} | acc: {acc:.2f}')
    
        model_path = f'{path_body}.{l+1}{path_ext}'
        joblib.dump([w, transition, tags], model_path)
        print_debug(f'this model is saevd to {model_path}')
    
def print_debug(msg):
    print(msg)

def load_data(doc=sys.stdin):
    *data, = load_train_data(doc)
    transition, tags = learn_data(data)

    return data, transition, tags

def load_train_data(doc):
    for line in doc:
        word_tag_list = [i.split('_') for i in line.strip().split()]
        *words, = [i[0] for i in word_tag_list]
        *tags, = [i[1] for i in word_tag_list]
        yield (words, tags)

def learn_data(data):
    transition = defaultdict(int)
    tags = set()
    tags.add('<s>')
    tags.add('</s>')

    for X, Y in data:
        # transition
        transition[f'<s> {Y[0]}'] += 1
        for i in range(1, len(Y)):
            transition[f'{Y[i-1]} {Y[i]}'] += 1
        transition[f'{Y[-1]} </s>'] += 1
        # tags
        for y in Y:
            tags.add(y)
    
    return transition, tags

def update_dict(w, items, amount=1):
    for k, v in items:
        w[k] += amount * v

def create_features(X, Y):
    phi = defaultdict(int)
    for i in range(len(Y) + 1):
        first_tag = "<s>" if i == 0 else Y[i-1]
        next_tag = "</s>" if i == len(Y) else Y[i]
        update_dict(phi, create_trans(first_tag, next_tag))
    for x, y in zip(X, Y):
        update_dict(phi, create_emit(y, x))
    return phi

def create_emit(tag, word):
    yield (f'E|{tag}|{word}', 1)

def create_trans(first_tag, next_tag):
    yield (f'T|{first_tag}|{next_tag}', 1)

def hmm_viterbi(w, X, transition, possible_tags):
    best_score, best_edge = {}, {}
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None

    # print(X)

    for i, x in enumerate(chain(X, ['</s>'])):
        for prev_tag in possible_tags:
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
        # print(best_edge)
    
    next_edge = f'{len(X)+1} </s>'
    tags = []
    while next_edge != None:
        idx, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]
    
    tags.reverse()
    # print(tags)
    return tags[1:-1]    

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')