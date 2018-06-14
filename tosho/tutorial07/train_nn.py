import sys, os
sys.path.append(os.pardir)
from dataset import titles_en_train
import numpy as np
from common.networks import Trainer, SGD, SimpleNeuralNetwork
from collections import defaultdict
from itertools import islice
import pickle

class FeatureModel:
    def __init__(self, feature_size, mode='train'):
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

        # unigram (up to 24,636 features)
        for token in x:
            features[self.idx['UNI: ' + token]] += 1
        
        # # bigram (up to 113,303 features)
        # for token in zip(x, x[1:]):
        #     features[self.idx[f'BI: {token[0]} {token[1]}']] += 1
        
        return features


def main():
    pkl_file = 'titles-en-train.labeled.pkl'
    fm_pkl_file = 'titles-en-train.labeled.fm.pkl'

    if os.path.isfile(pkl_file):
        print('load train data from pkl file ... ', end='')
        with open(pkl_file, 'rb') as f:
            x_train, t_train = pickle.load(f)
    else:
        print('load train data from source file ... ', end='')
        x_train, t_train = titles_en_train.load_labeled()
        extractor = FeatureModel(feature_size=25000)
        for i, x in enumerate(x_train):
            x_train[i] = extractor.extract(x)
        for i, t in enumerate(t_train):
            t_train[i] = 0 if t == -1 else 1

        with open(pkl_file, 'wb') as f:
            pickle.dump((x_train, t_train), f)
        with open(fm_pkl_file, 'wb') as f:
            pickle.dump(extractor.idx, f)

    x_train, t_train = np.array(x_train[:-1000]), np.array(t_train[:-1000])
    x_test, t_test = np.array(x_train[-1000:]), np.array(t_train[-1000:])
    print('Done')
    print(x_train.shape)
    print(t_train.shape)

    model = SimpleNeuralNetwork(x_train.shape[1], 64, 2)
    optimizer = SGD(0.1)
    trainer = Trainer(model, optimizer)

    print(f'Initial dev acc: {model.accuracy(x_test, t_test)}')
    trainer.train(x_train, t_train, max_epoch=3, batch_size=100)

    final_acc = model.accuracy(x_test, t_test)
    print(f'Final dev acc: {final_acc}')

    with open(f'model_{final_acc}.pkl', 'wb') as f:
        pickle.dump(model.params, f)

if __name__ == '__main__':
    main()