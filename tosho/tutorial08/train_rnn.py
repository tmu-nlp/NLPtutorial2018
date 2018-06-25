'''
python train_nn.py
python test_nn.py > ans.labeled
../../script/grade-prediction.py ../../data/titles-en-test.labeled ans.labeled
'''

import sys, os
sys.path.append(os.pardir)
from dataset import wiki_en_train
import numpy as np
from common.networks import Trainer, SGD, Momentum, SimpleRNN
from collections import defaultdict
from itertools import islice
import pickle

class FeatureModel:
    def __init__(self, size=25000, training=True):
        self.idx = defaultdict(self.new_id)
        self.size = size
        self.training = training

        # assign 0 to <unk>
        self.idx['<unk>'] = 0
        self.rev_idx = None

    def new_id(self):
        '''
        新しい素性用のidを発行する(train)。
        <unk>に相当するid(0)を発行する（dev、test）。
        '''
        if self.training:
            return len(self.idx)
        else:
            return 0

    def extract_phi(self, x):
        phi = []
        if isinstance(x, str):
            x = [x]
        for token in x:
            token = token.lower()
            phi.append(self.idx[token])
        return phi

    def learn(self, x):
        self.extract_phi(x)

    def one_hot(self, x):
        if isinstance(x, str):
            features = [0] * self.size
            for phi in self.extract_phi(x):
                features[phi] += 1
            return features
        else:
            phi = []
            for t in x:
                phi.append(self.one_hot(t))
            return phi

    def id2phi(self, id):
        if self.rev_idx == None:
            self.rev_idx = defaultdict(lambda:'<unk>')
            for key, value in self.idx.items():
                self.rev_idx[value] = key
        return self.rev_idx[id]

    def save_params(self, file_name='feature_model.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump((len(self.idx), self.idx), f)

    @staticmethod
    def load_params(file_name='feature_model.pkl'):
        with open(file_name, 'rb') as f:
            size, idx = pickle.load(f)
            model = FeatureModel(size=size, training=False)
            model.idx = idx
            model.idx.default_factory = model.new_id
            return model

def main():
    hidden_size = 64
    len_size = None
    data_size = None
    lr = 0.05
    batch_size = 10
    epochs = 50

    model_dir = f'model_h{hidden_size}_lr{lr}_b{batch_size}_e{epochs}'

    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    # load data
    print('load train data from source file... ', end='', flush=True)

    x_train, t_train = wiki_en_train.load_norm_pos()
    if data_size != None:
        x_train = x_train[:data_size]
        t_train = t_train[:data_size]

    # shrink sequence length
    if len_size != None:
        for i, X, T in zip(range(len(x_train)), x_train, t_train):
            X, T = X[:min(len_size, len(X))], T[:min(len_size, len(T))]
            x_train[i] = X
            t_train[i] = T

    print(f'{len(x_train)} sentences')

    # learn word/pos models
    print('learning language model... ', end='', flush=True)

    word_model = FeatureModel(training=True)
    for x in x_train: word_model.learn(x)

    pos_model = FeatureModel(training=True)
    for t in t_train: pos_model.learn(t)

    print(f'{len(word_model.idx)} tokens, {len(pos_model.idx)} pos')

    word_model.save_params(f'{model_dir}/model.word.pkl')
    pos_model.save_params(f'{model_dir}/model.pos.pkl')
    print('word and pos models are saved to model.word.pkl and model.pos.pkl')

    # training model
    word_model = FeatureModel.load_params(f'{model_dir}/model.word.pkl')
    pos_model = FeatureModel.load_params(f'{model_dir}/model.pos.pkl')

    model = SimpleRNN(word_model.size, hidden_size, pos_model.size)
    optimizer = Momentum(lr=lr)
    trainer = Trainer(model, optimizer,model_dir=model_dir)

    # embedding data
    for i, X, T in zip(range(len(x_train)), x_train, t_train):
        for j, x, t in zip(range(len(X)), X, T):
            X[j] = word_model.one_hot(x)
            # index vec として取り出す
            T[j] = pos_model.extract_phi(t)[0]
    x_train, t_train = np.array(x_train), np.array(t_train)
    trainer.train(x_train, t_train, max_epoch=epochs, batch_size=batch_size)

if __name__ == '__main__':
    main()