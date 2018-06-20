# -*- coding: utf-8 -*-
import numpy as np


class Output:

    def __init__(self):
        self.phi = 0
        self.loss = np.array([])

    def insert(self, phi):
        self.phi = phi

    def back_nn(self, ans):
        delta2 = (self.phi - ans) * (1 - self.phi ** 2)
        return delta2

    def insert_loss(self, phi, ans):
        delta2 = (phi - ans) * (1 - phi ** 2)
        self.loss = np.append(self.loss, delta2)

    def delta2(self):
        delta2 = np.average(self.loss)
        self.loss = np.array([])
        print(delta2)
        return delta2

    def result(self):
        if self.phi >= 0:
            return 1
        else:
            return -1


class Hidden:

    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.phi = np.array([])

    def insert_phi(self, phi):
        self.phi = phi

    def forward_nn(self):
        next_phi = np.tanh(np.dot(self.phi, self.w) + self.b)
        return next_phi

    def back_nn(self, delta2):
        delta = np.dot(delta2, self.w.T) * (1 - self.phi**2)
        return delta

    def update(self, delta2, lam):
        self.w -= lam * np.outer(delta2, self.phi).T
        self.b -= lam * delta2


class Input:

    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.phi = np.array([])

    def insert_phi(self, phi):
        self.phi = phi

    def forward_nn(self):
        next_phi = np.tanh(np.dot(self.w, self.phi) + self.b)
        return next_phi

    def update(self, delta, lam):
        self.w -= lam * np.outer(delta, self.phi)
        self.b -= lam * delta
