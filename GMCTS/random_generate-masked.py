import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from chainer import Variable

import nltk
import copy
import zinc_grammar
import cfg_util

from math import *
import numpy as np

import os
import sys

np.random.seed(0)
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

class RNN(chainer.Chain):
    def __init__(self, rule_size, emb_dim=128, hidden_dim=256):
        super(RNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(rule_size, emb_dim)
            self.lstm1 = L.LSTM(emb_dim, hidden_dim)
            self.out = L.Linear(hidden_dim, rule_size)
    
    def reset_state(self):
        self.lstm1.reset_state()
    
    def forward(self, x):
        h = self.embed(x)
        h = self.lstm1(h)
        h = self.out(h)
        return h

    def get_probability(self, x):
        h = self.forward(x)
        return F.softmax(h)

    def __call__(self, x, t):
        h = self.forward(x)
        return F.softmax_cross_entropy(h, t)

from rdkit.Chem import AllChem as Chem
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return smiles != '' and mol is not None

def main():
    rnn = RNN(rule_size=len(zinc_grammar.GCFG.productions()))
    serializers.load_npz("model-9.npz", rnn)
    rule_size = len(zinc_grammar.GCFG.productions())
    valid_smiles = []
    for trial in range(10000):
        print(trial, file=sys.stderr)
        rules_sampled = [0]
        stack = ['chain']
        rnn.reset_state()
        for _ in range(280):
            rule_prev = np.array([rules_sampled[-1]]).astype(np.int32)
            with chainer.no_backprop_mode():
                unmasked_prob = rnn.get_probability(rule_prev).data[0]
            if len(stack) > 0:
                p = stack.pop()
            else:
                p = 'Nothing'
            next_nonterminal = zinc_grammar.lhs_map[p]
            mask = zinc_grammar.masks[next_nonterminal]
            masked_prob_unnormalized = unmasked_prob * mask
            Z = np.sum(masked_prob_unnormalized)
            masked_prob = masked_prob_unnormalized / Z
            rule_sampled = np.random.choice(rule_size, p=masked_prob)
            rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal) 
                                    and (str(a) != 'None'),
                                    zinc_grammar.GCFG.productions()[rule_sampled].rhs())
            stack.extend(list(map(str, rhs))[::-1])
            rules_sampled.append(rule_sampled)
        smiles = cfg_util.decode(rules_sampled)
        if is_valid_smiles(smiles):
            valid_smiles.append(smiles)
            print(smiles, file=sys.stderr)
        if trial%100 == 0:
            print("{},{},{}".format(trial, len(valid_smiles), len(set(valid_smiles))))

if __name__ == "__main__":
    main()
