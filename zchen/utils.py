from pickle import load, dump
from collections import Counter, defaultdict
from math import log

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
    '''Turn file into n-gram count with affixes'''
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
    '''Turn counting into probability.'''
    total = sum(count.values())
    sos = "<%s>" % s_tok
    if prefix_count:
        prefix_total = sum(prefix_count.values())
        return total, { k: prefix_count[k[1:]]/prefix_total
                           if k[-2] == sos else
                           v/prefix_count[k[:-1]]
                       for k, v in count.items() }
    else:
        return total, { k: v/total for k, v in count.items() }


def witten_bell_weights(ngram_count):
    '''A special use case of \'defaultdict\''''
    type_variation_count = defaultdict(lambda:[0,set()])
    for ngram, count in ngram_count.items():
        t = type_variation_count[ngram[0]]
        t[0] += count
        t[1].add(ngram[1:])
    return {w:t[0]/(t[0]+len(t[1])) for w, t in type_variation_count.items()}


def unigram_smooth_gen(mod_prob, total_num_types):
    '''Special funtion for unigram.'''
    return lambda p: interpolate_gen(mod_prob)(p, 1 / total_num_types)


def interpolate_gen(main_prob):
    '''Produce fixed functions for interpolation, for the sake of computation'''
    if 0 == main_prob:
        return lambda p1, p2: p2
    elif 1 == main_prob:
        return lambda p1, p2: p1
    elif 0 < main_prob < 1:
        yet_prob = 1 - main_prob
        return lambda p1, p2: main_prob * p1 + yet_prob * p2
    raise ValueError("Invalid main weight.")


def _check_oov_call(oov_func):
    '''Validator for counting oov at test.'''
    def wrapper(*args):
        if args[0]._N_Gram__iv_counter is None:
            msg = "Calling '%s' before 'prof_of'" % oov_func.__name__
            msg += ", or lacking count = True in 'prob_of'"
            raise ValueError(msg)
        return oov_func(*args)
    return wrapper


class N_Gram:
    def __init__(self, n, model_name):
        self._model_file = model_name + ".%d" % n
        self._n_gram     = n
        self.__iv_counter = None
        self._oov_counter = None

    def build_model(self, train_file):
        with open(train_file, 'r') as fr:
            return count_n_gram_from_file(self._n_gram, fr)

    def seal_model(self, count_dict, prefix_count_dict = None):
        '''When adding more parameters, save them in the tuple for sealing them.
        Save them in the form (elem1, elem2,) + another_tuple. Don\'t forget to delete.'''
        witten_bell = witten_bell_weights(count_dict) if self._n_gram > 1 else None
        self._model = (self._n_gram, witten_bell) + cond_prob(count_dict, prefix_count_dict)
        with open(self._model_file, "wb") as fw:
            dump(self._model, fw)

    def load(self):
        with open(self._model_file, "rb") as fr:
            model = load(fr)
        if model[0] == self._n_gram and len(model) == 4:
            self._model = model
        else:
            raise ValueError("Trying to load unmatched %d-gram file." % model[0])

    @property
    def num_gram(self):
        return self._model[0]

    def weight_of(self, t):
        w = self._model[1].get(t, 0.0)
        return w, 1-w

    @property
    def num_tokens(self):
        return self._model[-2]

    @property
    def num_types(self):
        return len(self._model[-1].keys())

    def __str__(self):
        model = self._model[-1]
        s = "%d-gram model based on %d tokens in %s types"
        s += " with Witten-Bell weights\n" if self.num_gram > 1 else '\n'
        s = s % (self.num_gram, self.num_tokens, self.num_types)
        ml = max(sum(len(ngi)+2 for ngi in ng) for ng in model.keys())
        lop = "%-{}s%f\n".format(ml)
        for k, v in sorted(model.items(), key = lambda x:x[1]):
            s += lop % (", ".join(k), v)
        if self.num_gram > 1:
            s += "\t- Witten Bell weights:\n"
            for k,w in self._model[1].items():
                s += "\tw%-15s" % ("'" + k + "'") + '%f\n' % w
        return s

    def prob_of(self, test_file, count_oov = False):
        '''Should test the file with conditional chain.
        Multiply the yielded value or add log-linearly.'''
        # num_tokens is useless
        n, _, _, model = self._model
        sos, eos = make_padding(n - 1)
        self.__iv_counter = Counter() if count_oov else None
        self._oov_counter = Counter() if count_oov else None

        with open(test_file, 'r') as fr:
            for line in fr:
                lot = sos + line.strip().split() + eos
                for ng in n_gram(n, lot):
                    tok = ng[-1]
                    if ng in model:
                        if count_oov: self.__iv_counter[ng] += 1
                        yield tok, model[ng]
                    else:
                        if count_oov: self._oov_counter[ng] += 1
                        yield tok, 0.0

    @property
    @_check_oov_call
    def recent_oov(self):
        return self._oov_counter

    @property
    @_check_oov_call
    def recent_coverage_by_token(self):
        in_vocab = sum(self.__iv_counter.values())
        return in_vocab / (in_vocab + sum(self._oov_counter.values()))

    @property
    @_check_oov_call
    def recent_coverage_by_type(self):
        in_vocab = len(self.__iv_counter)
        return in_vocab / (in_vocab + len(self._oov_counter))

class N_Gram_Family:
    def __init__(self, max_n, family_name):
        self._family = [N_Gram(i, family_name) for i in range(1, max_n + 1)]

    def __getitem__(self, index):
        return self._family[index]

    def __len__(self):
        return len(self._family)

    def __str__(self):
        return "----\n".join(str(s) for s in self._family)

    def seal_model(self, train_file):
        '''The condition probability of N-Gram model stand on (N-1)-Gram model.
        Seal the models from 1-gram to n-gram'''
        pre_count = None
        for model in self._family:
            count = model.build_model(train_file)
            model.seal_model(count, pre_count)
            pre_count = count

    def load(self):
        for model in self._family:
            model.load()

    def entropy_of(self, test_file, weights, num_types_include_oov, base = None):
        '''About weights:
            When weights is None, apply Witten-Bell smooth to each word;
            When weights is a list of fixed numbers for each model, apply them to each model.'''
        assert weights is None or len( weights ) == len( self._family )
        if weights:
            assert all( 0 <= w <= 1 for w in weights)
            weights = tuple( interpolate_gen(w) for w in weights )
        else:
            weights    = (None,) * len(self._family)
        family     = tuple( model.prob_of(test_file) for model in self._family )
        _log       = (lambda p: log(p, base)) if base else log
        num_token  = 0
        log_prob   = 0
        processing = True
        while processing: # for a token in each n-gram model
            w_prob_by_last_model = 1 / num_types_include_oov # Zero-Gram
            for model, model_gen, weight in zip(self._family, family, weights):
                try:
                    w, w_prob = next(model_gen)
                except StopIteration:
                    processing = False
                    break
                if weight:
                    inter_prob = weight(w_prob, w_prob_by_last_model)
                    log_prob  += _log(inter_prob)
                elif model.num_gram > 1:
                    wb, cwb    = model.weight_of(w)
                    inter_prob = wb * w_prob + cwb * w_prob_by_last_model
                    log_prob  += _log(inter_prob)
                w_prob_by_last_model = w_prob
            num_token += 1
        return log_prob / num_token

if __name__ == "__main__":
    bigram_count = {('Tottori', 'is'):2, ('Tottori', 'city'):1}
    print(witten_bell_weights(bigram_count))
