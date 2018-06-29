from random import randint, random
from collections import defaultdict
from math import log2

def load_test_data(file_name='../../test/07-train.txt'):
    with open(file_name, 'r') as data:
        for i, line in enumerate(data):
            line = line.rstrip()
            for word in line.split():
                yield [i, word]

def load_wiki_data(file_name='../../data/wiki-en-documents.word'):
    with open(file_name, 'r') as data:
        for i, line in enumerate(data):
            line = line.rstrip()
            for word in line.split():
                yield [i, word]

def main(load_data=load_test_data, max_iter=10, topics=2):
    # initialize
    data = [[w[0], w[1], randint(0, topics-1)] for w in load_data()]
    xcounts, ycounts = defaultdict(int), defaultdict(int)

    words = set()
    for w in data:
        add_counts(xcounts, ycounts, w, 1)
        words.add(w[1])
    
    nw = len(words)
    print_interval = max(1, max_iter // 10)
    for t in range(max_iter):
        ll = 0
        for w in data:            
            add_counts(xcounts, ycounts, w, -1)
            probs = get_probs(xcounts, ycounts, w, nw, topics=topics, a=t+1, b=t+1)
            new_k = sample_one(probs)
            ll += log2(probs[new_k])
            w[2] = new_k
            add_counts(xcounts, ycounts, w, 1)

        if t % print_interval == 0:
            print(f'iter {t}|ll={ll:.2f}')
    
    for w in data:
        print(*w[1:])

def add_counts(xcounts, ycounts, w, amount):
    xcounts[f'{w[2]}'] += amount
    xcounts[f'{w[1]}|{w[2]}'] += amount
    ycounts[f'{w[0]}'] += amount
    ycounts[f'{w[2]}|{w[0]}'] += amount

def get_probs(xcounts, ycounts, w, nw, topics, a=1, b=1):
    probs = []
    for k in range(topics):
        c_k = xcounts[f'{k}']
        c_x_k = xcounts[f'{w[1]}|{k}']
        p_x_k = (c_x_k + a) / (c_k + a*nw)
        
        c_y = ycounts[f'{w[0]}']
        c_k_y = ycounts[f'{k}|{w[0]}'] 
        p_k_y = (c_k_y + b) / (c_y + b*topics)

        probs.append(p_x_k * p_k_y)
    return probs

def sample_one(probs):
    z = sum(probs)
    r = random()*z
    for k in range(len(probs)):
        r -= probs[k]
        if r <= 0:
            return k
    raise Exception()

if __name__ == '__main__':
    main(load_data=load_wiki_data, max_iter=10, topics=5)
    # import cProfile
    # cProfile.run('main()')