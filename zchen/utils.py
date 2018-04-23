from pickle import load, dump
s_tok = "s"


def n_gram(n, s):
    ns = []
    for i in range(n):
        ns.append(s[i:])
    return tuple(zip(*ns))


def make_padding(n):
    # this maker is xml. so single ending should appended
    sos = ["<%s>"  % s_tok] * n
    eos = ["</%s>" % s_tok]
    return sos, eos


def count_n_gram_from_file(n, fr):
    from collections import defaultdict
    count_dict = defaultdict(lambda : 0)
    # make padding
    sos, eos = make_padding( n-1 )

    for line in fr:
        # apply padding for list of tokens
        lot = line.strip().split(" ")
        lot = sos + lot + eos if s_tok else lot
        for token in n_gram(n, lot):
            count_dict[token] += 1
    return count_dict


def seal_model(count_dict):
    total = sum(count_dict.values())
    return total, { k: v/total for k, v in count_dict.items() }


class N_Gram:
    def __init__(self, n, model_name):
        self.__model_file = model_name + ".%d" % n
        self.__n_gram     = n

    def build_model(self, train_file):
        with open(train_file, 'r') as fr:
            count_dict = count_n_gram_from_file(self.__n_gram, fr)
        self.__model = (self.__n_gram,) + seal_model(count_dict)
        with open(self.__model_file, "wb") as fw:
            dump(self.__model, fw)

    def load(self):
        with open(self.__model_file, "rb") as fr:
            model = load(fr)
        if model[0] == self.__n_gram:
            self.__model = model
        else:
            raise ValueError("Trying to load unmatched N gram.")

    @property
    def num_gram(self):
        return self.__model[0]

    @property
    def num_tokens(self):
        return self.__model[1]

    @property
    def num_types(self):
        return len(self.__model[2].keys())

    def __repr__(self):
        s = "%d-gram model based on %d tokens in %s types\n"
        s = s % (self.num_gram, self.num_tokens, self.num_types)
        ml = max(sum(len(ngi) for ngi in ng) for ng in self.__model[2].keys())
        lop = "%-" + str(ml) + "s%f\n"
        for k, v in sorted(self.__model[2].items(), key = lambda x:x[1]):
            s += lop % (", ".join(k), v)
        return s

    def log_prob_of(self, test_file, oov_prob_count, base = None):
        from math import log
        n, _, model = self.__model      # num_tokens is useless
        oov_prob, oov_count = oov_prob_count
        sos, eos = make_padding(n - 1)

        total_num_types = self.num_types + oov_count
        oov_term = oov_prob * ( 1.0 / total_num_types )
        test_set_log_prob = 0.0
        with open(test_file, 'r') as fr:
            for line in fr:
                lot = sos + line.strip().split() + eos
                for ng in n_gram(n, lot):
                    ng_prob = model[ng] if ng in model else 0.0
                    ng_prob = ( 1 - oov_prob ) * ng_prob + oov_term
                    test_set_log_prob += log(ng_prob, base) if base else log(ng_prob)
                    # print(ng, ng_prob, ">", ( 1 - oov_prob ) * ng_prob + oov_term)
        return test_set_log_prob

if __name__ == "__main__":
    print(n_gram(1, "I am an NLPer"))
    print(n_gram(2, "I am an NLPer"))

    import sys
    with open(sys.argv[1], 'r') as fr:
        count_dict = count_n_gram_from_file(2, fr)
        for pair in seal_model(count_dict).items():
            print( *pair )
