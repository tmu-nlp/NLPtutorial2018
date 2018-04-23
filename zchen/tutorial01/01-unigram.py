from utils import count_n_gram_from_file, seal_model, make_padding, n_gram
from pickle import load, dump
import argparse
import sys
from math import log

def arguments_parse():
    parser = argparse.ArgumentParser(
        description='Mode(train, test, None), file(src, dst)',
        add_help=True,
    )
    parser.add_argument('-n', '--ngram', type=int, default=1)
    parser.add_argument('-m', '--mode', help='None mode is for view model with -s', type=str, choices=["train", "test"])
    parser.add_argument('-s', '--source', help='text for training or model for viewing', type=str)
    parser.add_argument('-d', '--destine', help='model.n to store or test text', type=str)
    #parser.add_argument('-l', '--log', help='use log probabilities', action='store_true', default=False)
    return parser.parse_args()

def build_model(src_file, n, save_to = None):
    with open(src_file, 'r') as fr:
        count_dict = count_n_gram_from_file(n, fr)
        num_tokens, model = seal_model(count_dict)
        if save_to:
            save_to = save_to + ".%d" % n
            print("Save to", save_to)
            with open(save_to, "wb") as fw:
                dump((n, num_tokens, model), fw)
        else:
            print("Printing the model instead of saving:")
            for key, value in model.items():
                print(", ".join(key), "\t", value)

def go_with(prob_model, test_file, oov_prob_count):
    from math import log, exp
    n, _, model = prob_model        # num_tokens is useless
    num_types   = len(model.keys()) # num_types is used
    oov_prob, oov_count = oov_prob_count

    total_num_types = num_types + oov_count
    oov_term = oov_prob * ( 1.0 / total_num_types )
    test_set_log_prob = 0.0
    sos, eos = make_padding(n - 1)
    with open(test_file, 'r') as fr:
        for line in fr:
            lot = sos + line.strip().split() + eos
            for ng in n_gram(n, lot):
                ng_prob = model[ng] if ng in model else 0.0
                ng_prob = ( 1 - oov_prob ) * ng_prob + oov_term
                test_set_log_prob += log(ng_prob, 2)
                # print(ng, ng_prob, ">", ( 1 - oov_prob ) * ng_prob + oov_term)
    print("Test set '%s'" % test_file, "\nLog probability:", test_set_log_prob)
    H = -test_set_log_prob / num_types # I still wonder why is not total_num_types!!
    print("Entropy:", H) # match the answer
    print("Perplexity: ", 2**H)

if __name__ == "__main__":
    args = arguments_parse()
    if args.mode == "train":
        build_model(args.source, args.ngram, args.destine)
    elif args.mode == "test":
        if args.source:
            with open(args.source, "rb") as fr:
                model = load(fr)
            go_with(model, args.destine, (0.05, 999995))
    else:
        with open(args.source, "rb") as fr:
            model = load(fr)
        print(args.source, model)
