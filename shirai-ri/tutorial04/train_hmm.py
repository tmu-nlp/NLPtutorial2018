
# coding: utf-8

# In[7]:

import codecs
from collections import defaultdict


def make_model(input_file, model_file):
    
    emit = defaultdict(int)
    transition = defaultdict(int)
    context = defaultdict(int)
    
    with codecs.open(input_file, 'r', 'utf8') as i_f, codecs.open(model_file, 'w', 'utf8') as m_f:
        for line in i_f:
            prev = '<s>'
            context[prev] += 1
            wordtags = line.strip().split()
            for wordtag in wordtags:
                word, tag = wordtag.split('_')
                transition[prev+' '+tag] += 1
                context[tag] += 1
                emit[tag+' '+word] += 1
                prev = tag
            transition[prev+' </s>'] += 1
            
        for key, value in transition.items():
            prev, tag = key.split(' ')
            m_f.write('T {} {}\n'.format(key, value/context[prev]))
        
        for key, value in emit.items():
            tag, word = key.split(' ')
            m_f.write('E {} {}\n'.format(key, value/context[tag]))


if __name__ == '__main__':
    make_model('./nlptutorial-master/data/wiki-en-train.norm_pos', './model_file.txt')
    # sys.argv[1], sys.argv[2]
    


