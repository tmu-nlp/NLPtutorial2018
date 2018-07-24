import sys
sys.path.append('..')
from utils.hmm_percep import *
from tqdm import tqdm

if '__main__' == __name__:
    hmm = HiddenMarkov(structed = True)
    for epoch in tqdm(range(5), desc = 'Training'):
        for o_s_gen in load_o_s_gen():
            hmm.add(o_s_gen)
    for i, lot in enumerate(load_text("../../data/wiki-en-test.norm")):
        hidden, verbose = hmm.get_hidden(lot, verbose = True)
        print(' '.join(hidden))
        with open(f"tsv/{i}.tsv" , "w") as fw:
            fw.write(verbose)
