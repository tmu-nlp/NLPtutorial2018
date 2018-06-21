'''
python train_nn.py
python test_nn.py > ans.labeled
../../script/grade-prediction.py ../../data/titles-en-test.labeled ans.labeled
'''

import sys, os
sys.path.append(os.pardir)
from dataset import titles_en_train
import numpy as np
from common.networks import Trainer, SGD, Momentum, SimpleNeuralNetwork
from collections import defaultdict
from itertools import islice
import pickle
from stemming.porter2 import stem

class FeatureModel:
    def __init__(self, feature_size=25000, mode='train'):
        self.idx = defaultdict(self.new_id)
        self.feature_size = feature_size
        self.mode = mode
    
    def new_id(self):
        '''
        新しい素性用のidを発行する。
        もし、素性サイズの上限を超える場合はunkとして、一律でself.feature_size - 1を割り当てる
        '''
        if self.mode == 'train':
            return min(len(self.idx), self.feature_size - 1)
        else:
            return self.feature_size - 1
        
    def extract(self, x):
        features = [0] * self.feature_size

        # unigram (up to 21,920 features)
        for token in x:
            token = token.lower()
            token = stem(token)
            features[self.idx['UNI: ' + token]] += 1
        
        # # bigram (up to 113,303 features)
        # for token in zip(x, x[1:]):
        #     features[self.idx[f'BI: {token[0]} {token[1]}']] += 1
        
        return features

    def save_params(self, file_name='feature_model.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(([*self.idx.items()], self.feature_size), f)
        
    def load_params(self, file_name='feature_model.pkl'):
        with open(file_name, 'rb') as f:
            idx, self.feature_size = pickle.load(f)
        for key, value in idx:
            self.idx[key] = value

def main():
    feature_size = 25000
    hidden_size = 512
    output_size = 2

    print('load train data from source file ... ')
    x_train, t_train = titles_en_train.load_labeled()

    feature_model = FeatureModel(feature_size=feature_size)

    for i, x in enumerate(x_train):
        x_train[i] = feature_model.extract(x)
    for i, t in enumerate(t_train):
        t_train[i] = 0 if t == -1 else 1

    print(f'{len(feature_model.idx)} features')
    feature_model.save_params()

    x_train, t_train = np.array(x_train[:-1000]), np.array(t_train[:-1000])
    x_test, t_test = np.array(x_train[-1000:]), np.array(t_train[-1000:])
    print(x_train.shape)
    print(t_train.shape)

    print('Start training ... ')
    model = SimpleNeuralNetwork(x_train.shape[1], hidden_size, output_size)
    optimizer = Momentum(lr=0.1)
    trainer = Trainer(model, optimizer)

    print(f'Initial dev acc: {model.accuracy(x_test, t_test)}')
    trainer.train(x_train, t_train, max_epoch=10, batch_size=100)

    final_acc = model.accuracy(x_test, t_test)
    print(f'Final dev acc: {final_acc}')

    with open(f'model.pkl', 'wb') as f:
        pickle.dump((feature_size, hidden_size, output_size, model.params), f)

if __name__ == '__main__':
    main()