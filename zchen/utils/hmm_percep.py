from collections import defaultdict as ddict
from typing import Dict, Tuple
from n_gram import unigram_smooth_gen, interpolate_gen, nlog_gen, n_gram
from data import load_o_s_gen, load_text
from itertools import chain
from heapq import nlargest

sos = '<s>'
eos = '</s>'

_matInt   = Dict[Tuple[str, str], int]
_matFloat = Dict[Tuple[str, str], float]

def _seal(jnt: _matInt, mgn: _matInt) -> _matFloat:
    prob = {}
    for from_to, jnt_cnt in jnt.items():
        prob[from_to] = jnt_cnt / mgn[from_to[0]]
    return prob

def _str(mt: _matFloat, prefix: str) -> str:
    s = ''
    for (old, new), prob in mt.items():
        s += f"{prefix} p('{new}'|'{old}') = {prob}\n"
    return s

def create_tran(old_state, state):
    yield ("T", old_state, state)

def create_emit(state, out):
    yield ("E", state, out)
    if out.istitle():
        yield ("F", 'CAPS', state)
    if out.endswith('ed'):
        yield ("F", 'SUF..ed', state)
    elif out.endswith('ing'):
        yield ("F", 'SUF..ing', state)

def create_features(states, outs):
    *emits, = zip(states, outs)
    trans = n_gram(2, [sos] + states)
    for t in trans:
        for f in create_tran(*t):
            yield f
    for t in emits:
        for f in create_emit(*t):
            yield f

def project(weights, gamma):
    score = 0
    tran, emit = n_gram(2, gamma)
    for feat in chain(create_tran(*tran), create_emit(*emit)):
        score += weights[feat]
    return score

class HiddenMarkov:
    '''Bayes' rule is invisible. But it is there as the cost.'''

    def __init__(self, **params):
        self._beam_size = params.get('beam_size', None)
        if params.get('structed', True):
            self._feature_weights = ddict(int)
        else:
            self._feature_weights = None
            self._trans = ddict(int)
            self._emits = ddict(int)
            self._states = ddict(int)
            self._outputs = ddict(int)
            self._uni_smooth = params.get('uni_smooth', unigram_smooth_gen(0.95, 1/1000000))
            self._bi_smooth  = params.get('bi_smooth',  interpolate_gen(0.95))

    def add(self, o_s_gen):
        old_state = sos
        if self._feature_weights is None: # generative
            self._states[sos] += 1
            for out, state in o_s_gen:
                self._outputs[out] += 1
                self._states[state] += 1
                # create_emit with perceptron
                self._emits[next(create_emit(state, out))] += 1
                # emit end
                self._trans[next(create_trans(old_state, state))] += 1
                old_state = state
            self._trans[(old_state, eos)] += 1
            self._states[eos] += 1 # cancel?
        else: # structed
            outs, gold_states = [], []
            self._states = ddict(int)
            for out, state in o_s_gen:
                outs.append(out)
                gold_states.append(state)
                self._states[state] += 1
            pred_states = self.get_hidden(outs, beam_size = self._beam_size)
            clear = all(po == go for po, go in zip(pred_states, gold_states))
            if not clear:
                self._update_feature_weights(create_features(gold_states, outs), 1)
                self._update_feature_weights(create_features(pred_states, outs), -1)

    def _update_feature_weights(self, feats, inc):
        for f in feats:
            self._feature_weights[f] += inc

    def seal(self):
        self._trans = _seal(self._trans, self._states)
        self._emits = _seal(self._emits, self._states)

    def __str__(self):
        return _str(self._trans, 'T') + _str(self._emits, 'E')

    def tran(self, states):
        if states in self._trans:
            return self._uni_smooth(self._trans[states])
        return self._uni_smooth(0)

    def emit(self, state_output):
        state, output = state_output
        if state_output in self._emits:
            word_uni_gram = self._uni_smooth(self._outputs[output])
            return self._bi_smooth(self._emits[state_output], word_uni_gram)
        return self._uni_smooth(0)

    def get_hidden(self, outputs, beam_size = None, verbose = False):
        if self._feature_weights is None:
            make_cost = nlog_gen()

        # forward lattice
        lattice = [{(None, sos):0}] # trans cost of 0 from None to sos
        for i, out in enumerate(outputs):
            layer = {}
            for new_state, prob in self._states.items(): # trans to
                if new_state == sos:
                    continue
                best_trans = None
                for last_trans, prev_cost in lattice[i].items(): # trans from
                    last_state = last_trans[-1]
                    if self._feature_weights is None:
                        trans_cost = make_cost(self.tran((last_state, new_state)))
                        emit_cost  = make_cost(self.emit((new_state, out)))
                        total_cost = trans_cost + emit_cost + prev_cost
                    else:
                        total_cost = prev_cost + project(self._feature_weights, (last_state, new_state, out))
                    if best_trans is None or total_cost < best_trans[-1]:
                        best_trans = (last_state, new_state), total_cost
                layer[best_trans[0]] = best_trans[-1] # from, to + cost
            if beam_size:
                beam = nlargest(beam_size, layer, key = lambda trans: layer[trans])
                layer = {trans:layer[trans] for trans in beam}
            lattice.append(layer)

        if verbose:
            verbose = 'postag\tid\tseq\tcost\n'
            for i, layer in enumerate(lattice):
                if i == 0:
                    continue
                min_val = min(layer.values())
                val_ran = max(layer.values()) - min_val
                for states, cost in layer.items():
                    verbose += '%s\t%d\t%s\t%f\n' % (states[-1], i, outputs[i-1], ((cost - min_val) / val_ran)**0.5)

        # backward lattice
        lattice.reverse()
        best_trans = min(lattice.pop(0).items(), key = lambda x: x[-1])[0]
        best_hidden = []
        while len(lattice): # sos at 0
            last_state, new_state = best_trans
            best_hidden.append(new_state)
            for last_trans in lattice.pop(0).keys():
                if last_trans[1] == last_state:
                    best_trans = last_trans
                    break
        best_hidden.reverse()

        if verbose:
            return best_hidden, verbose
        return best_hidden


if '__main__' == __name__:
    hmm = HiddenMarkov(structed = True)
    # print(hmm)
    for o_s_gen in load_o_s_gen():
        hmm.add(o_s_gen)
    hmm.seal()
    # print(hmm)
    for i, lot in enumerate(load_text("../../data/wiki-en-test.norm")):
        hidden, verbose = hmm.get_hidden(lot, verbose = True)
        print(' '.join(hidden))
        with open(f"tsv/{i}.tsv" , "w") as fw:
            fw.write(verbose)
