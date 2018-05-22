import os, sys
sys.path.append(os.path.pardir)

from collections import defaultdict
import pickle

def load_data(filename, mode='train'):
    '''
    return
    =====

    [
        [(word, pos),...],  # line 1
        [...],              # line 2
        ...
    ]
    '''
    with open(filename, 'r') as f:
        data = []
        for line in f:
            pairs = line.strip('\n').split(' ')
            pairs.insert(0, '<s>_BOS')
            pairs.append('</s>_EOS')

            this_data = []
            for pair in pairs:
                if mode == 'train':
                    word, pos = pair.split('_')
                    this_data.append((word, pos))
                elif mode == 'test':
                    word = pair.split('_')[0]
                    this_data.append(word)
            data.append(this_data)
        return data

def defaultdict_int():
    return defaultdict(int)
            
def iterate_n_gram(seq, n=2):
    for gram in zip(*[seq[i:] for i in range(n)]):
        yield gram

class PosModel:
    def __init__(self):
        self.Pt = None
        self.Pe = None
        self.lam = None
        self.unk_rate = None

    def train(self, data, vocab_size=10**6, lam=0.95):
        '''
        parameter[data]
        =====
        [
            [(word, pos),...],  # line 1
            [...],              # line 2
            ...
        ]
        '''

        self.Pt = self.__train_Pt(data)
        self.Pe = self.__train_Pe(data)
        self.lam = lam
        self.unk_rate = 1 / vocab_size
    
    def __train_Pt(self, data):
        pt = defaultdict(defaultdict_int)
        
        for line in data:
            poses = list(map(lambda p: p[1], line))
            for gram in iterate_n_gram(poses, 2):
                pt[gram[0]][gram[1]] += 1
        
        # frequency to probability
        for key, val in pt.items():
            s = sum(val.values())
            for subkey, subval in val.items():
                val[subkey] = subval / s
        
        return pt

    def __train_Pe(self, data):
        pe = defaultdict(defaultdict_int)
        
        for line in data:
            for pair in line:
                word = pair[0]
                pos = pair[1]

                pe[pos][word] += 1
        
        # frequency to probability
        for key, val in pe.items():
            s = sum(val.values())
            for subkey, subval in val.items():
                val[subkey] = subval / s
        
        return pe
    
    def save_params(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.Pt, self.Pe, self.lam, self.unk_rate), f)
    
    def load_params(self, filename):
        with open(filename, 'rb') as f:
            self.Pt, self.Pe, self.lam, self.unk_rate = pickle.load(f)

    def predict_pos(self, data):
        '''
        parameter[data]
        =====
        [
            [word, ...],    # line 1
            [...],          # line 2
            ...
        ]

        return
        =====
        [
            [pos, ...],     # line 1
            [...],          # line 2
            ...
        ]
        '''
        estimate = []
        for line in data:
            estimate.append(self.__predict_pos_line(line))
        return estimate
    
    def __predict_pos_line(self, line):
        for word in line:
            yield word
    
    def __vitabi_forward(self, line):
        best_edges = []
        best_scores = []


        return best_edges

if __name__ == '__main__':
    data = load_data('../../test/05-train-input.txt')
    print(data[:min(len(data), 10)])

'''
sample(test_train)
=====
a_X b_Y a_Z
a_X c_X b_Y

sample(test_data)
=====
a b a
a c b

sample(train_data)
=====
Natural_JJ language_NN processing_NN -LRB-_-LRB- NLP_NN -RRB-_-RRB- is_VBZ a_DT field_NN of_IN computer_NN science_NN ,_, artificial_JJ intelligence_NN -LRB-_-LRB- also_RB called_VBN machine_NN learning_NN -RRB-_-RRB- ,_, and_CC linguistics_NNS concerned_VBN with_IN the_DT interactions_NNS between_IN computers_NNS and_CC human_JJ -LRB-_-LRB- natural_JJ -RRB-_-RRB- languages_NNS ._.
'''
