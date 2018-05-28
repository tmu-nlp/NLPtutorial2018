from collections import defaultdict as ddict
from typing import Dict, Tuple

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

class Hidden_Markov:
    def __init__(self):
        self._trans = ddict(int)
        self._gen = ddict(int)
        self._states = ddict(int)
        self._outputs = ddict(int)

    def add(self, o_s_gen):
        old_state = sos
        self._states[sos] += 1
        for out, state in o_s_gen:
            self._outputs[out] += 1
            self._states[state] += 1
            self._gen[(state, out)] += 1
            self._trans[(old_state, state)] += 1
            old_state = state
        self._trans[(old_state, eos)] += 1
        self._states[eos] += 1 # cancel?

    def seal(self):
        self._trans = _seal(self._trans, self._states)
        self._gen = _seal(self._gen, self._states)

    def __str__(self):
        return _str(self._trans, 'T') + _str(self._gen, 'G')

    def get_hidden(self, outputs):
        from math import log
        make_cost = lambda x: -log(x) # deal with oov

        # forward lattice
        lattice = [{sos:0}]
        for i, out in enumerate(outputs):
            for new_state, prob in self._states:
                best_trans = None
                for last_state, prev_cost in lattice[i].items():
                    trans_cost = make_cost(self._trans[(last_state, new_state)])
                    emit_cost  = make_cost(self._gen[(new_state, out)])
                    total_cost = trans_cost + emit_cost + prev_cost
                    if best_trans and total_cost < best_trans[-1] or best_trans is None:
                            best_trans = (last_state, new_state), total_cost
                lattice.append(best_trans)

        # backward lattice
        lattice.reverse()
        lasttice.pop(0) # already in best_trans
        best_trans = best_trans[0]
        best_hidden = []
        while len(lattice) > 1: # sos at 0
            last_state, new_state = best_trans
            best_hidden.append(new_state)
            layer = lattice.pop(0)
            for last_trans in layer.keys():
                if last_trans[1] == last_state:
                    best_trans = last_trans
                    break
        best_hidden.reverse()
        return best_hidden


if '__main__' == __name__:
    hmm = Hidden_Markov()
    print(hmm)
    with open("../../test/05-train-input.txt") as fr:
        for line in fr:
            line = line.strip().split()
            o_s_gen = (o_s.split('_') for o_s in line)
            hmm.add(o_s_gen)
    hmm.seal()
    print(hmm)
