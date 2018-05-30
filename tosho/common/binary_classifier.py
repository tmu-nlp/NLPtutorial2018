import math
import numpy as np
from collections import defaultdict
from random import sample
import pickle
import matplotlib.pyplot as plt

class BinaryClassifier(object):

    def __init__(self):
        self.params = defaultdict(int)
        self.perceptron_threshold = 0.1
    
    def predict(self, x, verbose=False):
        features = self.__extract_feature(x)

        score = []
        for name, value in features.items():
            score.append(value * self.params[name])

        s = sum(score)

        if verbose:
            sorted_score = sorted(zip(score, features.keys()), key=lambda i: i[0])
            for item in sorted_score:
                print(f'{item[1]} -> {item[0]}')
            print('='*20)
            print(f'total : {s}')

        # return int(np.sign(s))
        if s > 0:
            return 1
        else:
            return -1

    def __extract_feature(self, x):
        f = defaultdict(int)
        # unigram
        for w in x:
            f['UNI: ' + w] += 1
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

    def gradient(self, x, t):
        features = self.__extract_feature(x)
        l = self.loss(x, t)

        for name, value in features.items():
            features[name] *= l
        
        return features

    def save_params(self, file_name='params.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)

class SimpleOptimizer(object):
    def __init__(self, lr=1):
        self.lr = lr
    
    def update(self, params, grads):
        for w, g in grads.items():
            new_val = params[w] + self.lr * g
            params[w] = new_val

class NormalizingOptimizer(object):
    def __init__(self, lr=1, thres=1000):
        self.lr = lr
        self.threas=5
    
    def update(self, params, grads):
        for w, g in grads.items():
            s = abs(g)
            if s > self.threas:
                g = self.threas * g / s
            
            new_val = params[w] + self.lr * g
            params[w] = new_val


class Trainer(object):
    def __init__(self, model, train_data, epochs=20,
                 optimizer=SimpleOptimizer()):
        self.model = model
        self.train_data = train_data
        self.epochs = epochs
        self.train_size = len(train_data)
        self.batch_size = self.train_size // self.epochs

        self.optimizer = optimizer

        self.train_acc_list = []
        self.dev_acc_list = []

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.__train_step(epoch)

        train_acc = np.average([self.model.accuracy(x, t) for x, t in self.train_data])

        print('=============== Final Dev Accuracy ===============')
        print(f'dev acc: {train_acc}')
        print(f'avg dev acc: {np.average(self.dev_acc_list):.2f} ({np.var(self.dev_acc_list):.4f})')
        

    def __train_step(self, epoch):
        dev_batch = self.train_data[(epoch - 1)*self.batch_size:epoch*self.batch_size]
        train_batch = self.train_data[:(epoch - 1)*self.batch_size] + self.train_data[epoch*self.batch_size:]

        # on-line learning
        for x, t in train_batch:
            grad = self.model.gradient(x, t)
            self.optimizer.update(self.model.params, grad)
        
        train_acc = np.average([self.model.accuracy(x, t) for x, t in train_batch])
        dev_acc = np.average([self.model.accuracy(x, t) for x, t in dev_batch])

        print(f'epoch {epoch} | train acc: {train_acc} | dev acc: {dev_acc}')

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