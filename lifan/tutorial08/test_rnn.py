from train_rnn import *
import pickle
import numpy as np

if __name__ == '__main__':
    with open('train_rnn_ids.dump', 'rb') as data_ids:
        x_ids, y_ids = pickle.load(data_ids)

    with open('train_rnn_network.dump', 'rb') as data_network:
        network = pickle.load(data_network)

    with open('../../data/wiki-en-test.norm', 'r') as data_test, open('my_answer.txt', 'w') as data_out:
        for line in data_test:
            answer = []
            x = []
            words = line.split()
            for word in words:
                if 'UNI:' + word not in x_ids.keys():
                    unk = np.zeros(len(x_ids))
                    x.append(unk)
                    continue
                x.append(CREATE_ONE_HOT(len(x_ids), x_ids['UNI:' + word]))

            h, p, y_pre = FORWARD_RNN(network, x)

            for each_y_pre in y_pre:
                answer.append(sorted(y_ids.items(), key=lambda x: x[1])[each_y_pre][0])
            print(' '.join(answer), file=data_out)