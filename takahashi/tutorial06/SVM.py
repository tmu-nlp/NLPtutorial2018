# -*- coding: utf-8 -*-
from collections import defaultdict
import pickle


class SVM:
    def __init__(self):
        self.model = defaultdict(int)
        self.iterations = 100
        self.margin = 20
        self.tfidf = None
        self.c = 0.0001

    def import_model(self, _model_file):
        with open(_model_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n', '')
                key, value = line.split('\t')
                self.model[key] = float(value)

    def save_model(self, _model_file):
        with open(_model_file, 'w', encoding='utf-8') as f:
            for key, value in self.model.items():
                f.write('{0}\t{1:.6f}\n'.format(key, value))

    def reset_model(self):
        self.model = defaultdict(int)

    def train(self, _input_file, _model_file=''):
        with open(_input_file, 'r', encoding='utf-8') as f:
            for itr in range(self.iterations):
                for line in f:
                    line = line.strip()
                    y, x = line.split('\t')
                    y = float(y)
                    phi = self.create_features(x)
                    _, score = self.predict_one(phi)
                    val = score * y
                    if val <= self.margin:
                        self.update_weights(phi, y)

        if _model_file != '':
            self.save_model(_model_file)

    def predict_all(self, _model_file, _input_file, _output_file):
        self.import_model(_model_file)

        f_out = open(_output_file, 'w', encoding='utf-8')

        with open(_input_file, 'r', encoding='utf-8') as f:
            for line in f:
                phi = self.create_features(line)
                y_prime, _ = self.predict_one(phi)
                ret = '{0}\t{1}'.format(y_prime, line)
                f_out.write(ret)
        f_out.close()

    def predict_one(self, phi):
        score = 0.0
        for name, value in phi.items():
            if name in self.model:
                score += value * self.model[name]
        if score >= 1:
            return 1, score
        else:
            return -1, score

    def update_weights(self, phi, y):
        def sign(x):
            if x < 0:
                return -x
            elif x > 0:
                return x
            return 0

        for name, value in self.model.items():
            if abs(value) < self.c:
                self.model[name] = 0
            else:
                self.model[name] -= sign(value) * self.c
        for name, value in phi.items():
            self.model[name] += value * y

    def create_features(self, x):
        return self.default_features(x)

    def default_features(self, x):
        phi = defaultdict(int)
        words = x.strip(). split(' ')
        for word in words:
            phi['UNI:' + word] += 1
        return phi

if __name__ == '__main__':
    svm = SVM()

    # Accuracy = 93.198725%
    print('train with default setting')
    input_file = '../../data/titles-en-train.labeled'
    model_file = '03-train-input-txt.model'
    svm.train(input_file, model_file)
    input_file = '../../data/titles-en-test.word'
    output_file = 'my_answer.labeled'
    svm.predict_all(model_file, input_file, output_file)
    svm.reset_model()
