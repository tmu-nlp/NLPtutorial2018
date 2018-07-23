# -*- coding: utf-8 -*-
from shift_reduce import Word, Queue, Stack, Feat, Score
import pickle
from tqdm import tqdm


def shift_reduce(queue):
    score = pickle.load(open('score.pkl', 'rb'))
    stack = Stack()
    stack.shift(queue.queue[0])
    queue.queue.pop(0)
    while len(queue.queue) > 0 or len(stack.stack) > 1:
        feat = Feat()
        feat.create_feats(stack, queue)
        score.shift_reduce(feat, stack, queue, pred=False)
    stack.result.append(stack.stack.pop(-1))
    return sorted(stack.result, key=lambda x: x.id)


def shift_reduce_train(queue, score):
    stack = Stack()
    stack.shift(queue.queue[0])
    queue.queue.pop(0)
    while len(queue.queue) > 0 or len(stack.stack) > 1:
        feat = Feat()
        feat.create_feats(stack, queue)
        pred = score.shift_reduce(feat, stack, queue, pred=True)
        corr = stack.calc_corr(queue)

        if pred != corr:
            score.update_weight(pred, corr, feat)


def train(path):
    queue = Queue()
    score = Score()
    epoch = 5
    for e in tqdm(range(epoch)):
        for i, line in enumerate(open(path, 'r')):
            if line == '\n':
                queue.calc_unproc()
                shift_reduce_train(queue, score)
                queue = Queue()
                continue
            line = line.strip().split('\t')
            queue.create(Word(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7]))
    pickle.dump(score, open('score.pkl', 'wb'))


def test(path):
    queue = Queue()
    with open('result', 'w') as f:
        for i, line in tqdm(enumerate(open(path, 'r'))):
            if line == '\n':
                queue.calc_unproc()
                result = shift_reduce(queue)
                print_file(f, result)
                queue = Queue()
                continue
            line = line.strip().split('\t')
            queue.create(Word(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7]))


def print_file(f, result):
    for r in result[1:]:
        print(
            f'{r.id}\t{r.word}\t{r.base}\t{r.pos}\t{r.pos2}\t{r.extend}\t{r.head_pred}\t{r.label}',
            file=f
        )
    print(file=f)


if __name__ == '__main__':
    train('../../data/mstparser-en-train.dep')
    test('../../data/mstparser-en-test.dep')
    # train('train')
