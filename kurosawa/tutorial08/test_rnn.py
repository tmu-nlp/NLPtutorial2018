import pickle
from train_rnn import *


if __name__ == '__main__':
    test_file = '../../data/wiki-en-test.norm'
    answer_file = 'my_answer.pos'
    ids_x_file = 'ids_x_file.byte'
    ids_y_file = 'ids_y_file.byte'
    net_file = 'weight_file.byte'
    with open(ids_x_file,'rb') as ids_x_data, open(ids_y_file,'rb') as ids_y_data, open(net_file,'rb') as net_data:
        ids_x = pickle.load(ids_x_data)
        ids_y = pickle.load(ids_y_data)
        net = pickle.load(net_data)
#making ids list (ID to predict)
    ids_y = {v:k for k,v in ids_y.items()}
    with open(test_file) as test, open(answer_file,'w') as answer:
        for line in test:
            #create one-hot vector
            x_list = []
            words = line.split()
            for word in words:
                x_list.append(create_one_hot(word.lower(), ids_x))
            #predict
            h,p,y_list = forward_rnn(net,x_list)
            ans = []
            for i in y_list:
                ans.append('{}_{}'.format(words.pop(0),ids_y[i]))
            answer.write(' '.join(ans))
            answer.write('\n')

