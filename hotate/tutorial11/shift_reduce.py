# -*- coding: utf-8 -*-
from collections import defaultdict


class Word:
    def __init__(self, id_, word, base, pos, pos2, extend, head_ans, label):
        self.id = int(id_)
        self.word = word
        self.base = base
        self.pos = pos
        self.pos2 = pos2
        self.extend = extend
        self.head_pred = 0
        self.head_ans = int(head_ans)
        self.label = label
        self.unproc = 0


class Queue:
    def __init__(self):
        self.queue = []

    def create(self, word):
        self.queue.append(word)

    def calc_unproc(self):
        for queue in self.queue:
            if queue.head_ans != 0:
                self.queue[queue.head_ans-1].unproc += 1


class Stack:
    def __init__(self):
        self.stack = [Word(0, 'ROOT', 'ROOT', -1, 0,0,0,0)]
        self.result = []

    def shift(self, word):
        self.stack.append(word)

    def reduce_left(self):
        self.stack[-2].head_pred = self.stack[-1].id
        self.result.append(self.stack.pop(-2))

    def reduce_right(self):
        self.stack[-1].head_pred = self.stack[-2].id
        self.result.append(self.stack.pop(-1))

    def shift_reduce(self, queue, mode):
        if mode == 'shift':
            self.shift(queue.queue[0])
            queue.queue.pop(0)
        elif mode == 'left':
            self.reduce_left()
        else:
            self.reduce_right()

    def calc_corr(self, queue):
        if self.stack[-1].head_ans == self.stack[-2].id and self.stack[-1].unproc == 0:
            self.reduce_right()
            self.stack[-1].unproc -= 1
            return 'right'
        elif self.stack[-2].head_ans == self.stack[-1].id and self.stack[-2].unproc == 0:
            self.reduce_left()
            self.stack[-1].unproc -= 1
            return 'left'
        else:
            self.shift(queue.queue[0])
            queue.queue.pop(0)
            return 'shift'


class Feat:
    def __init__(self):
        self.feats = defaultdict(int)

    def create_feats(self, stack, queue):
        if len(stack.stack) > 1:
            self.make_phi(stack.stack[-2], stack.stack[-1])
        if len(queue.queue) > 0:
            self.make_phi(stack.stack[-1], queue.queue[0])

    def make_phi(self, word_1, word_2):
        self.feats[f'{word_1.word}|{word_2.word}'] += 1
        self.feats[f'{word_1.word}|{word_2.pos}'] += 1
        self.feats[f'{word_1.pos}|{word_2.word}'] += 1
        self.feats[f'{word_1.pos}|{word_2.pos}'] += 1


class Score:
    def __init__(self):
        self.w_shift = defaultdict(int)
        self.w_left = defaultdict(int)
        self.w_right = defaultdict(int)
        self.w = defaultdict(int)

        self.shift = 0
        self.left = 0
        self.right = 0

    def initialize(self):
        self.shift = 0
        self.left = 0
        self.right = 0

    def calc_shift(self, feat):
        for key, value in feat.feats.items():
            self.shift += self.w_shift[key] * value

    def calc_left(self, feat):
        for key, value in feat.feats.items():
            self.left += self.w_left[key] * value

    def calc_right(self, feat):
        for key, value in feat.feats.items():
            self.right += self.w_right[key] * value

    def compare(self, stack, queue, pred):
        if len(stack.stack) > 1:

            if self.shift >= self.left and self.shift >= self.right and len(queue.queue) > 0:
                if pred:
                    return 'shift'
                else:
                    stack.shift(queue.queue[0])
                    queue.queue.pop(0)

            elif self.left >= self.right:
                if pred:
                    return 'left'
                else:
                    stack.reduce_left()

            else:
                if pred:
                    return 'right'
                else:
                    stack.reduce_right()
        else:
            stack.shift(queue.queue[0])
            queue.queue.pop(0)

    def shift_reduce(self, feat, stack, queue, pred):
        self.initialize()
        self.calc_shift(feat)
        self.calc_left(feat)
        self.calc_right(feat)
        return self.compare(stack, queue, pred)

    def update_shift(self, feat, mode):
        for key, value in feat.feats.items():
            if mode == 1:
                self.w_shift[key] += value
            else:
                self.w_shift[key] -= value

    def update_left(self, feat, mode):
        for key, value in feat.feats.items():
            if mode == 1:
                self.w_left[key] += value
            else:
                self.w_left[key] -= value

    def update_right(self, feat, mode):
        for key, value in feat.feats.items():
            if mode == 1:
                self.w_right[key] += value
            else:
                self.w_right[key] -= value

    def update_weight(self, pred, corr, feat):
        if pred == 'shift':
            if corr == 'right':
                self.update_shift(feat, 0)
                self.update_right(feat, 1)
            elif corr == 'left':
                self.update_shift(feat, 0)
                self.update_left(feat, 1)
        elif pred == 'left':
            if corr == 'shift':
                self.update_left(feat, 0)
                self.update_shift(feat, 1)
            elif corr == 'right':
                self.update_left(feat, 0)
                self.update_right(feat, 1)
        else:
            if corr == 'shift':
                self.update_right(feat, 0)
                self.update_shift(feat, 1)
            elif corr == 'left':
                self.update_right(feat, 0)
                self.update_left(feat, 1)
