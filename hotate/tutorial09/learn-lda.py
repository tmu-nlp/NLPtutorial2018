# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import random


def initialize(path, topic_size):
    x_corpus = []
    y_corpus = []

    x_counts = defaultdict(lambda: 0)
    y_counts = defaultdict(lambda: 0)
    vocab = {}

    for doc_id, line in enumerate(open(path, 'r')):
        words = line.lower().split()
        topics = []
        for word in words:
            topic = random.randrange(topic_size)
            topics.append(topic)
            add_counts(word, topic, doc_id, 1, x_counts, y_counts)
            vocab[word] = 1
        x_corpus.append(words)
        y_corpus.append(topics)

    return x_corpus, x_counts, y_corpus, y_counts, vocab


def add_counts(word, topic, doc_id, amount, x_counts, y_counts):
    x_counts[topic] += amount
    x_counts[f'{word}|{topic}'] += amount

    y_counts[doc_id] += amount
    y_counts[f'{topic}|{doc_id}'] += amount


def sampling(path, topic_size):
    x_corpus, x_counts, y_corpus, y_counts, vocab = initialize(path, topic_size)

    epoch = 8
    for e in range(epoch):
        ll = 0
        for doc_id in range(len(x_corpus)):
            for word_id in range(len(x_corpus[doc_id])):
                x = x_corpus[doc_id][word_id]
                y = y_corpus[doc_id][word_id]
                add_counts(x, y, doc_id, -1, x_counts, y_counts)
                probs = []
                for k in range(topic_size):
                    prob = x_given_k(x_counts, x, k, vocab) * k_given_y(y_counts, k, doc_id, topic_size)
                    probs.append(prob)
                new_y = sample_one(probs)
                ll += math.log(probs[new_y])
                add_counts(x, new_y, doc_id, 1, x_counts, y_counts)
                y_corpus[doc_id][word_id] = new_y
        print(ll)
    print_ans(x_corpus, y_corpus)


def print_ans(x_corpus, y_corpus):
    with open('my_answer', 'w') as f:
        for x_word, y_topic in zip(x_corpus, y_corpus):
            for x, y in zip(x_word, y_topic):
                f.write(f'{x}_{y} ')
            print(file=f)


def x_given_k(x_counts, x, k, vocab):
    alfa = 0.01
    prob = (x_counts[f'{x}|{k}'] + alfa) / (x_counts[k] + (alfa * len(vocab)))
    return prob


def k_given_y(y_counts, k, doc_id, topic_size):
    beta = 0.01
    prob = (y_counts[f'{k}|{doc_id}'] + beta) / (y_counts[doc_id] + (beta * topic_size))
    return prob


def sample_one(probs):
    z = sum(probs)
    remaining = random.uniform(0, z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i


if __name__ == '__main__':
    sampling('../../test/07-train.txt', 2)
    # sampling('../../data/wiki-en-documents.word', 10)
