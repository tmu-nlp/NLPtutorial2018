import sys
sys.path.append("..")
from utils.n_gram import N_Gram_Family
from utils.solutions import grid_search, arg_range
import argparse

def arguments_parse():
    parser = argparse.ArgumentParser(
        description='Mode(train, test, None), file(src, dst)',
        add_help=True,
    )
    parser.add_argument('-n', '--ngram', type=int, default=2)
    parser.add_argument('-m', '--mode', help='None mode is for view model with -s', type=str, choices=["train", "test"])
    parser.add_argument('-s', '--source', help='text for training or model for viewing', type=str)
    parser.add_argument('-d', '--destine', help='model.n to store or test text', type=str)
    #parser.add_argument('-l', '--log', help='use log probabilities', action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = arguments_parse()
    if args.mode == "train":
        model = N_Gram_Family(args.ngram, args.destine)
        model.build(args.source)
    elif args.mode == "test": # use family instead
        model = N_Gram_Family(args.ngram, args.source)
        model.load()
        model.seal()
        # model.prepare([0.95, 0.95], 1/1000000)
        # print("Test set '%s'" % args.destine)
        # print("Entropy:", model.entropy_of(args.destine)) # match the answer
        min_e, max_e = arg_range(grid_search(model, args.destine))
        fmt = "uni(%f), bi(%f), entropy(%f)"
        print(fmt % min_e)
        print(fmt % max_e)
        model.prepare(None, 1/1000000)
        print("Entropy with Witten-Bell smooth:", model.entropy_of(args.destine))
    else:
        model = N_Gram_Family(args.ngram, args.source)
        model.load()
        model.seal()
        print(model)
