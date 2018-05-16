import sys
sys.path.append("..")
from utils.n_gram import N_Gram, unigram_smooth_gen, nlog_gen
import argparse
_nlog = nlog_gen(2)

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
    elif args.mode == "test":
        model = N_Gram(1, args.source)
        model.load()
        model.seal()
        inp = unigram_smooth_gen(0.95, 1/1000000) # inp avoids log(0.0)
        nlog_prob = sum(_nlog(inp(p)) for _, p in model.cond_prob_of(args.destine, count_oov = True))

        print("Test set '%s'" % args.destine)
        print("Log probability:", nlog_prob)
        H = nlog_prob / model.num_types # I still wonder why is not total_num_types!!
        print("Entropy:        ", H) # match the answer
        print("Perplexity:     ", 2**H)
        coverage = (model.recent_coverage_by_token, model.recent_coverage_by_type)
        print("Coverage: %.2f by token, %.2f by type" % coverage)
    else:
        model = N_Gram(1, args.source)
        model.load()
        model.seal()
        print(model)
