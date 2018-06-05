import math
import numpy as np
from collections import defaultdict
from random import sample
import pickle
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)
from common.functions import sigmoid, d_sigmoid

class BinaryClassifier(object):

    def __init__(self):
        self.params = defaultdict(int)
        self.perceptron_threshold = 0.1
    
    def predict(self, x, verbose=False):
        features = self.__extract_feature(x)

        score = []
        for name, value in features.items():
            score.append(value * self.params[name])

        s = sigmoid(sum(score))

        if verbose:
            sorted_score = sorted(zip(score, features.keys()), key=lambda i: i[0])
            for value, name in sorted_score:
                print(f'{name} : sigmoid({features[name]} * {self.params[name]}) = {value}')
            print('='*20)
            print(f'total : {s}')

        if s >= 0.5:
            return 1
        else:
            return -1

    def __extract_feature(self, x):
        f = defaultdict(int)
        # unigram
        for w in x:
            f['UNI: ' + w] += 1
        # bigram
        for pair in zip(*[x[i:] for i in range(2)]):
            f[f'BI: {pair[0]} {pair[1]}'] += 1
        
        return f
    
    def loss(self, x, t):
        y = self.predict(x)
        if y == t:
            return 0
        else:
            return t

    def accuracy(self, x, t):
        y = self.predict(x)
        if t == y:
            return 1
        else:
            return 0

    def loss_margin_with_gradient(self, x, t):
        features = self.__extract_feature(x)
        l = self.loss(x, t)

        # gradient
        score = []
        for name, value in features.items():
            score.append(value * self.params[name])
        dW = d_sigmoid(sum(score))
        for name, value in features.items():
            features[name] = t * value * dW
        
        # margin
        m = sum(score) * t

        return l, m, features

    def save_params(self, file_name='params.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)

class SimpleOptimizer(object):
    def __init__(self, lr=0.1, thres=0.0001):
        self.lr = lr
        self.thres = thres

        self.current_iter = 0
        self.V = defaultdict(int)   # version of weights
    
    def update(self, params, grads):
        for name, grad in grads.items():
            current_value = params[name]

            # update weight by L1 norm
            if self.V[name] != self.current_iter:
                norm = self.thres * (self.current_iter - self.V[name])
                if abs(current_value) > norm:
                    current_value -= np.sign(current_value) * norm
                else:
                    current_value = 0
                self.V[name] = self.current_iter
            
            params[name] = current_value + grad

        self.current_iter += 1

class Trainer(object):
    def __init__(self, model, train_data, epochs=20,
                 optimizer=SimpleOptimizer(), margin_thres=15):
        self.model = model
        self.train_data = train_data
        self.epochs = epochs
        self.train_size = len(train_data)
        self.batch_size = self.train_size // self.epochs

        self.optimizer = optimizer
        self.margin_thres = margin_thres

        self.train_acc_list = []
        self.dev_acc_list = []

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.__train_step(epoch)

        train_acc = np.average([self.model.accuracy(x, t) for x, t in self.train_data])

        print('=============== Final Dev Accuracy ===============')
        print(f'dev acc: {train_acc:.2f}')
        print(f'avg dev acc: {np.average(self.dev_acc_list):.2f} ({np.var(self.dev_acc_list):.4f})')
        

    def __train_step(self, epoch):
        dev_batch = self.train_data[(epoch - 1)*self.batch_size:epoch*self.batch_size]
        train_batch = self.train_data[:(epoch - 1)*self.batch_size] + self.train_data[epoch*self.batch_size:]

        # on-line learning
        test_margin_list = []
        for x, t in train_batch:
            l, m, grad = self.model.loss_margin_with_gradient(x, t)
            test_margin_list.append(m)
            if l != 0 or m < self.margin_thres:
                self.optimizer.update(self.model.params, grad)
        
        train_acc = np.average([self.model.accuracy(x, t) for x, t in train_batch])
        dev_acc = np.average([self.model.accuracy(x, t) for x, t in dev_batch])
        test_margin = np.average(test_margin_list)

        print(f'epoch {epoch} | train acc: {train_acc:.2f} | dev acc: {dev_acc:.2f} | test margin: {test_margin:.2f}')

        self.train_acc_list.append(train_acc)
        self.dev_acc_list.append(dev_acc)


    def draw_accuracy(self, file_name='figure.png'):
        epochs = list(range(1, len(self.train_acc_list)+1))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_acc_list)
        plt.xlabel('epoch')
        plt.ylabel('train acc')
        plt.title(f'avg: {np.average(self.train_acc_list):.2f} var: {np.var(self.train_acc_list):.4f}')

        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.dev_acc_list)
        plt.xlabel('epoch')
        plt.ylabel('dev acc')
        plt.title(f'avg: {np.average(self.dev_acc_list):.2f} var: {np.var(self.dev_acc_list):.4f}')

        plt.subplot(2, 2, 3)
        plt.plot(self.train_acc_list, self.dev_acc_list)
        plt.xlabel('train acc')
        plt.ylabel('dev acc')

        fig = plt.gcf()
        plt.draw()
        fig.savefig(file_name, dpi=100)

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.pardir)
    from common.utils import load_labeled_data, load_word_data

    model = BinaryClassifier()

    train_data = list(load_labeled_data('../../test/03-train-input.txt'))
    test_data = list(load_word_data('../../test/03-train-answer.txt'))
    trainer = Trainer(model, train_data, test_data)

    trainer.train()
    model.save_params('test.pkl')

    model2 = BinaryClassifier()
    model2.load_params('test.pkl')

    print(model.params)
    print(model2.params)