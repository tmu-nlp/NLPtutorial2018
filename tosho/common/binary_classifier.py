import math
import numpy
from collections import defaultdict
from random import sample
import pickle
import matplotlib.pyplot as plt

class BinaryClassifier(object):
    def __init__(self):
        # alias to params['W']
        self.params = defaultdict(int)
    
    def predict(self, x):
        score = [self.params[word] for word in x]
        return numpy.sign(sum(score))
    
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
            params[w] += self.lr * g

class Trainer(object):
    def __init__(self, model, train_data, test_data,
                 epochs=20, mini_batch_size=100,
                 optimizer=SimpleOptimizer()):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.batch_size = min(mini_batch_size, len(train_data))
        self.train_size = len(train_data)
        self.iter_per_epoch = max(self.train_size // self.batch_size, 1)
        self.iter_cap = int(self.epochs * self.iter_per_epoch)

        self.optimizer = optimizer

        self.current_iter = 0
        self.current_epoch = 0

        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        for i in range(self.iter_cap):
            self.__train_step()

        test_acc = numpy.average([self.model.accuracy(x, t) for x, t in self.test_data])

        print('=============== Final Test Accuracy ===============')
        print(f'test acc: {test_acc}')

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
            test_sample = sample(self.test_data, min(len(self.test_data), self.batch_size))
            train_acc = numpy.average([self.model.accuracy(x, t) for x, t in train_sample])
            test_acc = numpy.average([self.model.accuracy(x, t) for x, t in test_sample])

            # print(f'epoch {self.current_epoch} | train acc: {train_acc}')
            print(f'epoch {self.current_epoch} | train acc: {train_acc} | test acc: {test_acc}')

            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

        self.current_iter += 1

    def draw_accuracy(self, file_name='figure.png'):
        plt.plot(self.train_acc_list, self.test_acc_list)
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