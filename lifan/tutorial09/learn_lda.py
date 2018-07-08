from collections import defaultdict
import random

def add_counts(word, topic, docid, amount):
    sentence_counts[topic] += amount
    sentence_counts[word + '|' + str(topic)] += amount
    topic_counts[docid] += amount
    topic_counts[str(topic) + '|' + str(docid)] += amount
    return sentence_counts, topic_counts

def sample_one(probs):
    z = sum(probs)
    remaining = random.random() * z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i

if __name__ == '__main__':
    sentence_corpus = []
    topic_corpus = []
    sentence_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    x_type = defaultdict(int)
    l = 10
    alpha = 0.01
    beta = 0.01

    for line in open('../../data/wiki-en-documents.word'):
        docid = len(sentence_corpus)
        words = line.strip().split()
        topics = []
        for word in words:
            topic = random.randint(0,l)
            topics.append(topic)
            add_counts(word, topic, docid, 1)
            x_type[word] += 1
        sentence_corpus.append(words)
        topic_corpus.append(topics)

    for epoch in range(l):
        for i in range(len(sentence_corpus)):
            for j in range(len(sentence_corpus[i])):
                x = sentence_corpus[i][j]
                y = topic_corpus[i][j]
                add_counts(x, y, i, -1)
                probs = []
                for k in range(l):
                    xprob = (sentence_counts[x + '|' + str(k)] + alpha) / (sentence_counts[k] + alpha * len(x_type))
                    yprob = (topic_counts[str(k) + '|' + str(i)] + beta) / (topic_counts[i] + beta * 10)
                    probs.append( xprob * yprob )
                new_topic = sample_one(probs)
                add_counts(x, new_topic, i, 1)
                topic_corpus[i][j] = new_topic
    lists = [set() for i in range(l)]    
    for x, y in zip(sentence_corpus, topic_corpus):
        for word, topic in zip(x, y):
            lists[topic].add(word)
    for i, words in enumerate(lists):
        print(i, list(words)[:10])

