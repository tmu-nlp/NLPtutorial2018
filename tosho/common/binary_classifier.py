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
        score = [self.params[word] for word in x]
        s = sum(score)

        if verbose:
            sorted_score = sorted(zip(score, x), key=lambda i: i[0])
            for item in sorted_score:
                print(f'{item[1]} -> {item[0]}')
            print('='*20)
            print(f'total : {s}')

        if abs(s) < self.perceptron_threshold:
            return 0
        else:
            return np.sign(s)
    
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
        l = self.loss(x, t)

        grads = defaultdict(int)
        for word in x:
            grads[word] += l
        
        return grads

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
            # # 正則化
            # if abs(new_val) >= 10:
            #     new_val = new_val / 10
            params[w] = new_val

class Trainer(object):
    def __init__(self, model, train_data,
                 epochs=20, mini_batch_size=100,
                 optimizer=SimpleOptimizer()):
        self.model = model
        self.train_data = train_data
        self.epochs = epochs
        self.batch_size = min(mini_batch_size, len(train_data))
        self.train_size = len(train_data)
        self.iter_per_epoch = max(self.train_size // self.batch_size, 1)
        self.iter_cap = int(self.epochs * self.iter_per_epoch)

        self.optimizer = optimizer

        self.current_iter = 0
        self.current_epoch = 0

        self.train_acc_list = []
        self.dev_acc_list = []

    def train(self):
        for i in range(self.iter_cap):
            self.__train_step()

        train_acc = np.average([self.model.accuracy(x, t) for x, t in self.train_data])

        print('=============== Final Dev Accuracy ===============')
        print(f'dev acc: {train_acc}')
        print(f'avg dev acc: {np.average(self.dev_acc_list):.2f} ({np.var(self.dev_acc_list):.4f})')
        

    def __train_step(self):
        train_batch = sample(self.train_data, self.batch_size)

        # on-line learning
        for x, t in train_batch:
            grad = self.model.gradient(x, t)
            self.optimizer.update(self.model.params, grad)
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            #TODO: use sampling
            train_sample = sample(self.train_data, self.batch_size)
            test_sample = sample(self.train_data, min(len(self.train_data), self.batch_size))
            train_acc = np.average([self.model.accuracy(x, t) for x, t in train_sample])
            test_acc = np.average([self.model.accuracy(x, t) for x, t in test_sample])

            # print(f'epoch {self.current_epoch} | train acc: {train_acc}')
            print(f'epoch {self.current_epoch} | train acc: {train_acc} | test acc: {test_acc}')

            self.train_acc_list.append(train_acc)
            self.dev_acc_list.append(test_acc)

        self.current_iter += 1

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
        plt.show()
        plt.draw()
        fig.savefig(file_name, dpi=100)

if __name__ == '__main__':
    import os, sys
    sys.path.append(os.pardir)
    from common.utils import load_labeled_data, load_word_data

    model = BinaryClassifier()

    train_data = list(load_labeled_data('../../test/03-train-input.txt'))
    test_data = list(load_word_data('../../test/03-train-answer.txt'))
    trainer = Trainer(model, train_data, test_data, mini_batch_size=1)

    trainer.train()
    model.save_params('test.pkl')

    model2 = BinaryClassifier()
    model2.load_params('test.pkl')

    print(model.params)
    print(model2.params)