import pickle
from train_nn import create_features,forward_nn

if __name__ == '__main__':
    with open('weight_file.txt','rb') as w, open('id_file.txt','rb') as id_:
        net = pickle.load(w)
        ids = pickle.load(id_)
    with open('../../data/titles-en-test.word') as test, open('my_answer.labeled','w') as answer:
        for line in test:
            phi0 = create_features(line,ids)
            phi = forward_nn(net,phi0)
            y = (1 if phi[len(net)-1][0] >= 0 else -1)
            answer.write('{}\n'.format(y))
