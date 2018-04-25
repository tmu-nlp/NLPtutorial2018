from pickle import load, dump

s_tok = "s"


def n_gram(n, s):
    ns = []
    for i in range(n):
        ns.append(s[i:])
    return tuple(zip(*ns))


def make_padding(n):
    # this maker is not xml. so single ending should appended
    sos = ["<%s>"  % s_tok] * n
    eos = ["</%s>" % s_tok]
    return sos, eos


def count_n_gram_from_file(n, fr):
    from collections import Counter
    count_dict = Counter()
    # make padding
    sos, eos = make_padding( n-1 )

    for line in fr:
        # apply padding for list of tokens
        lot = line.strip().split(" ")
        lot = sos + lot + eos if s_tok else lot
        count_dict.update(n_gram(n, lot))
    return count_dict


def cond_prob(count, prefix_count):
    total = sum(count.values())
    sos = "<%s>" % s_tok
    if prefix_count:
        # problem when cal <s><s>w in 3-gram
        prefix_total = sum(prefix_count.values())
        return total, { k: prefix_count[k[1:]]/prefix_total if k[-2] == sos else v/prefix_count[k[:-1]] for k, v in count.items() }
    else:
        return total, { k: v/total for k, v in count.items() }


def unigram_smooth_gen(mod_prob, total_num_types):
    if 0 < mod_prob < 1:
        oov_prob = 1 - mod_prob
        oov_term = oov_prob * ( 1.0 / total_num_types )
        return lambda p: mod_prob * p + oov_term
    if mod_prob == 0:
        return lambda p: oov_term
    elif mod_prob == 1:
        return lambda p: p
    raise ValueError("Invalid main weight.")


def interpolate_gen(main_prob):
    if 0 == main_prob:
        return lambda p1, p2: p2
    elif 1 == main_prob:
        return lambda p1, p2: p1
    elif 0 < main_prob < 1:
        yet_prob = 1 - main_prob
        return lambda p1, p2: main_prob * p1 + yet_prob * p2
    raise ValueError("Invalid main weight.")


class N_Gram:
    def __init__(self, n, model_name):
        self.__model_file = model_name + ".%d" % n
        self.__n_gram     = n

    def build_model(self, train_file):
        with open(train_file, 'r') as fr:
            return count_n_gram_from_file(self.__n_gram, fr)

    def seal_model(self, count_dict, prefix_count_dict = None):
        self.__model = (self.__n_gram,) + cond_prob(count_dict, prefix_count_dict)
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

    def prob_of(self, test_file, base = None):
        # num_tokens is useless
        n, _, model = self.__model
        sos, eos = make_padding(n - 1)

        with open(test_file, 'r') as fr:
            for line in fr:
                lot = sos + line.strip().split() + eos
                for ng in n_gram(n, lot):
                    yield model[ng] if ng in model else 0.0

class N_Gram_Family:
    def __init__(self, max_n, family_name, num_types_include_oov):
        self.__family = [N_Gram(i, family_name) for i in range(1, max_n + 1)]
        self.__total_types = num_types_include_oov

    def seal_model(self, train_file):
        pre_count = None
        for model in self.__family:
            count = model.build_model(train_file)
            model.seal_model(count, pre_count)
            pre_count = count

    # same thing here
    def log_prob_of(self, test_file, oov_prob_count, model_weights, base = None):
        assert sum(model_weights) == 1 and len(model_weights) == len[self.__family]
        return sum(m.log_prob_of(test_file, oov_prob_count, base) * w for m, w in zip(self.__family, model_weights))

if __name__ == "__main__":
    print(n_gram(1, "I am an NLPer"))
    print(n_gram(2, "I am an NLPer"))

    import sys
    with open(sys.argv[1], 'r') as fr:
        count_dict = count_n_gram_from_file(2, fr)
        for pair in seal_model(count_dict).items():
            print( *pair )
