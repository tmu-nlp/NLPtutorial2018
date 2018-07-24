import sys
sys.path.append('..')
from utils.hmm_percep import *
from tqdm import tqdm

if '__main__' == __name__:
    hmm = HiddenMarkov(structed = True, beam_size = 2)
    for _ in tqdm(range(10), desc = 'Training with beam%s' % 2):
        for o_s_gen in load_o_s_gen():
            hmm.add(o_s_gen)
    for i, lot in enumerate(load_text("../../data/wiki-en-test.norm")):
        hidden, verbose = hmm.get_hidden(lot, verbose = True, beam_size = 10)
        print(' '.join(hidden))
        with open(f"tsv/{i}.tsv" , "w") as fw:
            fw.write(verbose)
