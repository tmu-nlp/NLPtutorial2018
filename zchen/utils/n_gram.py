from pickle import load, dump
from collections import Counter, defaultdict
from math import log
from data import load_text

s_tok = "s"
_log_base = None
_second_change_log_base = False


def nlog_gen(new_base = None):
    global _log_base, _second_change_log_base
    if _second_change_log_base:
        raise Warning("Please pay attention to consistency!")
    if new_base and new_base != _log_base:
        _log_base = new_base
        _second_change_log_base = True
    return (lambda p: -log(p, _log_base)) if _log_base else (lambda x:-log(x))


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


def count_n_gram_from_file(n, fname):
    '''Turn file into n-gram count with affixes'''
    count_dict = Counter()
    # make padding
    sos, eos = make_padding( n-1 )

    for lot in load_text(fname):
        # apply padding for list of tokens
        lot = sos + lot + eos if s_tok else lot
        count_dict.update(n_gram(n, lot))
    return count_dict


def cond_prob(count, prefix_count):
    '''Turn counting into probability.'''
    total = sum(count.values())
    sos = "<%s>" % s_tok
    if prefix_count:
        prefix_total = sum(prefix_count.values())
        return { k: prefix_count[k[1:]]/prefix_total
                    if k[-2] == sos else
                    v/prefix_count[k[:-1]]
                for k, v in count.items() }
    else:
        return { k: v/total for k, v in count.items() }


def witten_bell_weights(ngram_count):
    '''A special use case of \'defaultdict\''''
    type_variation_count = defaultdict(lambda:[0,set()])
    for ngram, count in ngram_count.items():
        t = type_variation_count[ngram[0]]
        t[0] += count
        t[1].add(ngram[1:])
    return {w:t[0]/(t[0]+len(t[1])) for w, t in type_variation_count.items()}


def unigram_smooth_gen(mod_prob, zero_gram_prob):
    '''Special funtion for unigram.'''
    return lambda p: interpolate_gen(mod_prob)(p, zero_gram_prob)


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


    for k,w in weights.items():
        s += "\tw%-15s" % ("'" + k + "'") + '%f\n' % w
    return s


def _str(kv_dict, key_is_tuple):
    s = ''
    ml = max(sum(len(ngi)+2 for ngi in ng)
                if key_is_tuple else
             len(ng)+2 for ng in kv_dict.keys())
    lop = "\t%-{}s%f\n".format(ml)
    for k, v in sorted(kv_dict.items(), key = lambda x:x[1]):
        s += lop % (", ".join(k) if key_is_tuple else k, v)
    return s


class N_Gram:
    def __init__(self, n, model_name):
        self._model_file = model_name + ".%d" % n
        self._n_gram     = n
        self.__iv_counter = None
        self._oov_counter = None
        self._prob        = None
        self._cond_prob   = None
        self._paddings   = make_padding(n - 1)

    def build(self, train_file):
        count       = count_n_gram_from_file(self._n_gram, train_file)
        witten_bell = witten_bell_weights(count) if self._n_gram > 1 else None
        self._model = (self._n_gram, witten_bell, count)
        with open(self._model_file, "wb") as fw:
            dump(self._model, fw)
        return self

    def seal(self, prefix_count_dict = None):
        self._cond_prob = cond_prob(self.raw_count, prefix_count_dict)
        if self._n_gram == 1:
            self._prob = self._cond_prob

    def load(self):
        with open(self._model_file, "rb") as fr:
            model = load(fr)
        if model[0] == self._n_gram and len(model) == 3:
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
    def raw_count(self):
        return self._model[-1]

    @property
    def num_types(self):
        return len(self.raw_count)

    @property
    def num_tokens(self):
        return sum(self.raw_count.values())

    def make_padding(self, lot):
        sos, eos = self._paddings
        return sos + lot + eos

    def __getattr__(self, attr_name):
        if attr_name.startswith('iter'):
            attr_name = attr_name[4:]
            if attr_name == '_count':
                data = self._model[1]
            elif attr_name == '_witten_bell':
                data = self._model[-1]
            elif attr_name == '_prob':
                data = self._prob
            elif attr_name == '_cond_prob':
                data = self._cond_prob
        else:
            raise AttributeError("'NGram' object has no attribute '%s'" % attr_name)
        return data.items()

    def __str__(self):
        s = "%d-gram model based on %d tokens in %s types.\n"
        s = s % (self.num_gram, self.num_tokens, self.num_types)
        if self._prob:
            s += "- Joint probability:\n" + _str(self._prob, key_is_tuple = True)
        if self._cond_prob and self._n_gram > 1:
            s += "- Conditional probability:\n" + _str(self._cond_prob, key_is_tuple = True)
        if self.num_gram > 1:
            s += "- Witten Bell weights:\n" + _str(self._model[1], key_is_tuple = False)
        return s

    def prob_of(self, ng):
        if self._prob is None:
            self._prob = cond_prob(self.raw_count)
        if self._n_gram == 1 and isinstance(ng, tuple):
            return self._prob[ng[0]]
        return self._prob[ng]

    def cond_prob_of(self, test_file, count_oov = False):
        # num_tokens is useless
        n, model = self.num_gram, self._cond_prob
        self.__iv_counter = Counter() if count_oov else None
        self._oov_counter = Counter() if count_oov else None

        with open(test_file, 'r') as fr:
            for line in fr:
                lot = line.strip().split()
                for ng in n_gram(n, self.make_padding(lot)):
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
        self._range  = tuple(range(1, max_n + 1))
        self._family = [N_Gram(i, family_name) for i in self._range]

    def __str__(self):
        return "----\n".join(str(s) for s in self._family)

    def build(self, train_file):
        for model in self._family:
            model.build(train_file)

    def load(self):
        for model in self._family:
            model.load()

    def seal(self):
        '''The condition probability of N-Gram model stand on (N-1)-Gram model.
        Seal the models from 1-gram to n-gram'''
        pre_count = None
        for model in self._family:
            model.seal(pre_count)
            pre_count = model.raw_count

    def prepare(self, weights, zero_gram_prob):
        '''About weights:
            When weights is None, apply Witten-Bell smooth to each word;
            When weights is a list of fixed numbers for each model, apply them to each model.'''
        if weights and len( weights ) == len( self._family ):
            self._weights = tuple( interpolate_gen(w) for w in weights )
        elif weights is None:
            self._weights = (None,) * len(self._family)
        else:
            raise ValueError("Invalid weights, thought I can perform the crossing version!")
        self._zero_gram_prob = zero_gram_prob

    def entropy_of(self, test_file):
        family     = tuple( model.cond_prob_of(test_file) for model in self._family )
        _nlog      = nlog_gen()
        num_token  = 0
        log_prob   = 0
        processing = True
        while processing: # for a token in each n-gram model
            w_prob_by_last_model = self._zero_gram_prob
            for model, model_gen, weight in zip(self._family, family, self._weights):
                try:
                    w, w_prob = next(model_gen)
                except StopIteration:
                    processing = False
                    break
                if weight:
                    inter_prob = weight(w_prob, w_prob_by_last_model)
                elif model.num_gram > 1:
                    wb, cwb    = model.weight_of(w)
                    inter_prob = wb * w_prob + cwb * w_prob_by_last_model
                else: # mixture smooth Witten-Bell and others
                    inter_prob = w_prob
                w_prob_by_last_model += inter_prob
            log_prob  += _nlog(w_prob_by_last_model)
            num_token += 1
        return log_prob / num_token

    def cond_prob_of(self, ng):
        n = len[ng]
        family = self._family[n:]
        weights = self._weights[n:]
        uni_gram = family.pop(0)
        uni_weight = weights.pop(0)
        prob_ = uni_gram.prob_of(ng[-1])
        cond_prob_ = uni_weight(prob_, self._zero_gram_prob)
        smoothed = 0
        for weight, model in zip(weights, family):
            prob = model.prob_of(ng[-model.num_gram:])
            cond_prob = prob / prob_
            smoothed += weight(cond_prob, cond_prob_)
            prob_ = prob
            cond_prob_ = cond_prob
        return smooothed


if __name__ == "__main__":
    bigram_count = {('Tottori', 'is'):2, ('Tottori', 'city'):1}
    print(witten_bell_weights(bigram_count))
