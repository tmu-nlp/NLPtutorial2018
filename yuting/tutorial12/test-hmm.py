from train_hmm import *
import dill



if __name__ == '__main__':
    path_data_test = '../../data/wiki-en-test.norm'
    path_data_out = 'result/my_answer.txt'
    path_w_tags_in = 'result/w_p_t.dump'
    with open(path_w_tags_in, 'rb') as w_tags_in:
        w, possible_tags, transition = dill.load(w_tags_in)
    with open(path_data_test) as data_test, open(path_data_out, 'w') as data_out:
        for line in data_test:
            X = line.strip().split()
            Y_hat = hmm_viterbi(x)

            print(' '.join(Y_hat), file=data_out)