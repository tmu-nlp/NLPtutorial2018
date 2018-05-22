import os, sys
sys.path.append(os.path.pardir)

from collections import defaultdict

def load_data(filename):
    '''
    [
        # line 1
        [
            (word, pos),
            ...
        ],
        # line 2
        [
            ...
        ],
        ...
    ]
    '''
    with open(filename, 'r') as f:
        data = []
        for line in f:
            this_data = []
            for pair in line.strip('\n').split(' '):
                word, pos = pair.split('_')
                this_data.append((word, pos))
            data.append(this_data)
        return data
            
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
        self.Pt = self.__train_Pt(data)
        self.Pe = self.__train_Pe(data)
        self.lam = lam
        self.unk_rate = 1 / vocab_size
    
    def __train_Pt(self, data):
        pt = defaultdict(self.__create_defaultdict)
        
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
        pe = defaultdict(self.__create_defaultdict)
        
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
    
    def __create_defaultdict(self):
        '''
        defaualtdit 初期化のための関数
        '''
        return defaultdict(int)

if __name__ == '__main__':
    data = load_data('../../test/05-train-input.txt')
    print(data[:min(len(data), 10)])

'''
sample(test)
=====
a_X b_Y a_Z
a_X c_X b_Y

sample(data)
=====
Natural_JJ language_NN processing_NN -LRB-_-LRB- NLP_NN -RRB-_-RRB- is_VBZ a_DT field_NN of_IN computer_NN science_NN ,_, artificial_JJ intelligence_NN -LRB-_-LRB- also_RB called_VBN machine_NN learning_NN -RRB-_-RRB- ,_, and_CC linguistics_NNS concerned_VBN with_IN the_DT interactions_NNS between_IN computers_NNS and_CC human_JJ -LRB-_-LRB- natural_JJ -RRB-_-RRB- languages_NNS ._.
'''
