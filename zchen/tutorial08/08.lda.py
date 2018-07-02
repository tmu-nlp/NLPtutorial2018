from random import randint, random
from collections import defaultdict, Counter
from math import log2


wiki_path = '../../data/wiki-en-documents.word'
test_path = '../../test/07-train.txt'

topic_word_cnt = defaultdict(int)
doc_topic_cnt  = defaultdict(int)

def load_data(fname):
    with open(fname, 'r') as data:
        for i, line in enumerate(data):
            line = line.rstrip()
            for j, word in enumerate(line.split()):
                yield i, j, word

def train(data_path, num_topics, num_epochs=10):

    w2t = {wij:randint(0, num_topics -1) for wij in load_data(data_path)}
    vocab = Counter(w[-1] for w in w2t.keys())
    for wijt in w2t.items():
        add_counts(wijt, 1)

    num_words = len(vocab)
    for e in range(num_epochs):
        corpus_entropy = 0
        for wijt in w2t.items():
            add_counts(wijt, -1)
            probs = get_probs(wijt, num_words, num_topics)
            topic = sample_one(probs)
            corpus_entropy += -log2(probs[topic])
            w2t[wijt[0]] = topic
            add_counts((wijt[0], topic), 1)

        print(f'Epoch {e} entropy:{corpus_entropy:.2f}')

def add_counts(wijt, amount):
    (doc_id, word_pos, word), topic = wijt
    topic_word_cnt[topic] += amount
    topic_word_cnt[f'{word}|{topic}'] += amount
    doc_topic_cnt[doc_id] += amount
    doc_topic_cnt[f'{topic}|{doc_id}'] += amount

def get_probs(wijt, num_words, num_topics, a=1, b=1):
    probs = []
    (doc_id, word_pos, word), _ = wijt
    for topic in range(num_topics): # all topics for one word
        c_t   = topic_word_cnt[topic]
        c_t_w = topic_word_cnt[f'{word}|{topic}']
        p_t_w = (c_t_w + a) / (c_t + a * num_words)

        c_d   = doc_topic_cnt[doc_id]
        c_d_t = doc_topic_cnt[f'{topic}|{doc_id}']
        p_d_t = (c_d_t + b) / (c_d + b * num_topics)

        probs.append(p_d_t * p_t_w)
    return probs # pseudo-prob

def sample_one(probs):
    z = sum(probs)
    r = random()*z
    for k in range(len(probs)):
        r -= probs[k]
        if r <= 0:
            return k

if __name__ == '__main__':
    train(wiki_path, 20, num_epochs = 10)
