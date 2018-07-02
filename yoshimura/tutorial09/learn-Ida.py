import sys
import numpy as np
import random
from collections import defaultdict
import math

xcorpus = []  # 各x,yを格納
ycorpus = []  # カウントの格納
xcounts = defaultdict(lambda: 0)
ycounts = defaultdict(lambda: 0)
alpha = 0.02
beata = 0.02

data_path = '../../test/07-train.txt'

NUM_TOPICS = 2
EPOCH = 5
# NUM_TOPICS = int(sys.argv[1])
# EPOCH = int(sys.argv[2])


def add_counts(word, topic, doc_id, amount):
    xcounts[topic] += amount
    xcounts[f'{word}|{topic}'] += amount
    ycounts[doc_id] += amount
    ycounts[f'{topic}|{doc_id}'] += amount
    

def sample_one(probs):
    z = sum(probs)  # 確率の和を計算
    remaining = random.random() * z
    for i in range(len(probs)):  # probsの各項目を検証
        remaining -= probs[i]  # 現在の確率を引く
        if remaining <= 0:
            return i
    raise Exception('Error at sample_one')


def prob_topic_k(x, k, Y):
    Nx = len(xcorpus)
    Ny = len(ycorpus)
    P_x_given_k = (xcounts[f'{x}|{k}'] + alpha) / (xcounts[k] + alpha * Nx)
    P_y_given_Y = (ycounts[f'{k}|{Y}'] + beata) / (ycounts[Y] + beata * Ny)

    return P_x_given_k * P_y_given_Y


# 初期化
for line in open(data_path, 'r'):
    doc_id = len(xcorpus)  # この文章のIDを取得
    words = line.rstrip().split(' ')
    # topics = [random.randint(0, len(words)) for _ in range(len(words))]  # 単語のトピックをランダム初期化
    topics = []
    for word in words:
        topic = random.randint(0, NUM_TOPICS - 1)  # [0, NUM_TOP)の間
        topics.append(topic)
        add_counts(word, topic, doc_id, 1)
    xcorpus.append(words)
    ycorpus.append(topics)

# サンプリング
for _ in range(EPOCH):
    LL = 0
    for i in range(len(xcorpus)):  # 各文
        for j in range(len(xcorpus[i])):  # 各単語
            x = xcorpus[i][j]
            y = ycorpus[i][j]
            add_counts(x, y, i, -1)  # 各カウントの減算(-1)
            probs = []
            for k in range(NUM_TOPICS):
                probs.append(prob_topic_k(x, k, i))  # トピックkの確率
            new_y = sample_one(probs)
            LL += math.log(probs[new_y])  # 対数尤度の計算
            add_counts(x, new_y, i, 1)  # 各カウントの加算
            ycorpus[i][j] = new_y
    print(LL)

for i in range(len(xcorpus)):
    for j in range(len(xcorpus[i])):
        x = xcorpus[i][j]
        y = ycorpus[i][j]
        print(f'{x}_{y}', end=' ')
    print()