import sys
sys.path.append("..")
from utils.n_gram import N_Gram, nlog_gen, unigram_smooth_gen
from utils.solutions import viterbi, Trie
import argparse


def arguments_parse():
    parser = argparse.ArgumentParser(
        description='Mode(train, test, None), file(src, dst)',
        add_help=True,
    )
    parser.add_argument('-m', '--mode', help='None mode is for view model with -s', type=str, choices=["train", "test"])
    parser.add_argument('-s', '--source', help='text for training or model for viewing', type=str)
    parser.add_argument('-d', '--destine', help='model.n to store or test text', type=str)
    #parser.add_argument('-l', '--log', help='use log probabilities', action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = arguments_parse()
    if args.mode == "train":
        model = N_Gram(1, args.destine)
        model.build(args.source)
    elif args.mode == "test": # use family instead
        model = N_Gram(1, args.source)
        model.load()
        model.seal()
        prob_proc = unigram_smooth_gen(0.95, 1/100000000000)
        trie = Trie(model, prob_proc, nlog_gen())
        with open(args.destine) as fr:
            for ans in fr:
                dsp = ans.strip()
                ans = dsp.split(' ')
                que = ''.join(ans)
                print("Origin: %s\n split: %s\n" % ('|'.join(ans), '|'.join(viterbi(trie, que, verbose = True)))) # match the answer
    else:
        model = N_Gram(1, args.source)
        model.load()
        model.seal()
        print(model)
