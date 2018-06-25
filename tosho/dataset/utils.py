import sys

def load_labeled(file_name, lowercasing=True):
    if lowercasing:
        decorator = lambda w: w.lower()
    else:
        decorator = lambda w: w
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip('\n')
            t, x = line.split('\t')
            x = [decorator(word) for word in x.split(' ')]
            yield (x, int(t))

def load_norm_pos(file_name, lowercasing=True):
    '''
    '''
    '''
    Sample
    ==========
    Natural_JJ language_NN processing_NN -LRB-_-LRB- NLP_NN -RRB-_-RRB- is_VBZ a_DT field_NN of_IN computer_NN science_NN ,_, artificial_JJ intelligence_NN -LRB-_-LRB- also_RB called_VBN machine_NN learning_NN -RRB-_-RRB- ,_, and_CC linguistics_NNS concerned_VBN with_IN the_DT interactions_NNS between_IN computers_NNS and_CC human_JJ -LRB-_-LRB- natural_JJ -RRB-_-RRB- languages_NNS ._.
    '''
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip('\n')
            for x, t in [pair.split('_') for pair in line.split(' ')]:
                yield (x, t)
            yield ('</s>', '</s>')

def load_norm():
    for line in sys.stdin:
        yield line.rstrip().split(' ')