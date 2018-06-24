# -*- coding: utf-8 -*-
import numpy as np


class Output:

    def __init__(self):
        self.p = []
        self.y = []
        self.true_y = []

    def forward(self, hidden_out):
        pred = np.tanh(hidden_out)
        self.p.append(pred)

    def answer(self, y_ids):
        for p in self.p:
            p = p[0]
            y = -1
            for i in range(len(p)):
                if p[i] > p[y]:
                    y = i
            for key, y_id in y_ids.items():
                if y_id == y:
                    self.y.append(key)
                    break
        return self.y

    def backward(self, y, t):
        err_out = y - self.p[t]
        self.true_y.append(y)
        return err_out

    def initialize(self):
        self.p = []
        self.y = []
        self.true_y = []


class Hidden:

    def __init__(self, w_h, w_o, b_o):
        self.b_o = b_o
        self.w_h = w_h
        self.w_o = w_o

        self.h = []

        self.err_h = []

        self.delta_b_o = 0
        self.delta_w_h = 0
        self.delta_w_o = 0

    def forward(self, input_hidden, t):
        if t > 0:
            next_h = np.tanh(input_hidden + np.dot(self.h[t - 1], self.w_h))
        else:
            next_h = np.tanh(input_hidden)
        self.h.append(next_h)
        self.err_h.append(0)

        hidden_out = np.dot(self.h[t], self.w_o) + self.b_o
        return hidden_out

    def backward_delta(self, err_out, t):
        self.delta_w_o += np.outer(self.h[t], err_out)
        self.delta_b_o += err_out

    def backward(self, err_out, t):
        self.backward_delta(err_out, t)

        if t == len(self.h)-1:
            err_hidden = np.dot(err_out, self.w_o.T)
        else:
            err_hidden = np.dot(err_out, self.w_o.T) + np.dot(self.err_h[t+1], self.w_h.T)
        err_hidden = err_hidden * (1 - self.h[t]**2)
        self.err_h[t] = err_hidden

        if t != 0:
            self.delta_w_h += np.outer(self.h[t-1], err_hidden)

        return err_hidden

    def update(self, lam):
        self.w_h += lam * self.delta_w_h
        self.w_o += lam * self.delta_w_o
        self.b_o += lam * self.delta_b_o

    def initialize(self):
        self.h = []
        self.err_h = []
        self.delta_b_o = 0
        self.delta_w_h = 0
        self.delta_w_o = 0


class Input:

    def __init__(self, w, b):
        self.x = []
        self.w = w
        self.b = b

        self.delta_w = 0
        self.delta_b = 0

    def forward(self, x):
        self.x.append(x)
        input_hidden = np.dot(x, self.w) + self.b
        return input_hidden

    def backward(self, err_hidden, t):
        self.delta_w += np.outer(self.x[t], err_hidden)
        self.delta_b += err_hidden

    def update(self, lam):
        self.w += lam * self.delta_w
        self.b += lam * self.delta_b

    def initialize(self):
        self.x = []
        self.delta_w = 0
        self.delta_b = 0
