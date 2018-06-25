import sys, os
sys.path.append(os.pardir)
from common.networks import SimpleRNN
from tutorial08.train_rnn import FeatureModel
from dataset.utils import load_norm
import numpy as np

def main():
    model_dir = sys.argv[1]
    epoch = sys.argv[2]

    *model_file, = filter(lambda f: f.startswith(f'model_{epoch}'),  os.listdir(model_dir))
    model_file = f'{model_dir}/{model_file[0]}'
    word_model_file = f'{model_dir}/model.word.pkl'
    pos_model_file = f'{model_dir}/model.pos.pkl'

    model = SimpleRNN.load_params(model_file)
    word_model = FeatureModel.load_params(word_model_file)
    pos_model = FeatureModel.load_params(pos_model_file)

    for x in load_norm():
        phi = np.array(word_model.one_hot(x))
        y = model.predict(phi)
        y = np.argmax(y, axis=1)
        pos = [pos_model.id2phi(p).upper() for p in y]
        print(*[f'{p}' for t, p in zip(x, pos)])

if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()')