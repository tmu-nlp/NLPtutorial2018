import sys, os
sys.path.append(os.pardir)
from dataset import titles_en_test
from tutorial07.train_nn import FeatureModel
from common.networks import SimpleNeuralNetwork
from common.utils import load_word_data
import numpy as np
import pickle

def main():
    model_file = 'model.pkl'

    feature_model = FeatureModel(25000, mode='test')
    feature_model.load_params()

    model = None
    with open(model_file, 'rb') as f:
        input_size, hidden_size, output_size, params = pickle.load(f)

        model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
        for i, p in enumerate(params):
            model.params[i][...] = p

    for x in load_word_data('../../data/titles-en-test.word'):
        feature = np.array(feature_model.extract(x))
        y = model.predict(feature)
        y = 1 if y[1] > y[0] else -1
        
        print(f'{y}\t{" ".join(x)}')

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')