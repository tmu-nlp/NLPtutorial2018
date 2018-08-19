from train_hmm import hmm_viterbi
import dill



if __name__ == '__main__':
    path_data_test = '../../data/wiki-en-test.norm'
    path_data_out = 'result/my_answer.txt'
    path_w_tags_in = 'result/w_p_t.dump'
    with open(path_w_tags_in, 'rb') as w_tags_in:
        weight, possible_tags, dict_transition = dill.load(w_tags_in)
    with open(path_data_test) as data_test, open(path_data_out, 'w') as data_out:
        for line in data_test:
            x = line.strip().split()
            y_hat = hmm_viterbi(weight,x,possible_tags,dict_transition)

            print(' '.join(y_hat), file=data_out)