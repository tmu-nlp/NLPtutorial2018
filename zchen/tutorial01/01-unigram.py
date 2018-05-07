import sys
sys.path.append("..")
from utils import N_Gram, unigram_smooth_gen
import argparse
from math import log

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
        cnt = model.build_model(args.source)
        model.seal_model(cnt)
    elif args.mode == "test":
        model = N_Gram(1, args.source)
        model.load()
        inp = unigram_smooth_gen(0.95, 1000000)
        log_prob = sum(log(inp(p), 2) for p in model.prob_of(args.destine, 2))

        print("Test set '%s'" % args.destine, "\nLog probability:", log_prob)
        H = -log_prob / model.num_types # I still wonder why is not total_num_types!!
        print("Entropy:", H) # match the answer
        print("Perplexity: ", 2**H)
    else:
        model = N_Gram(1, args.source)
        model.load()
        print(model)
