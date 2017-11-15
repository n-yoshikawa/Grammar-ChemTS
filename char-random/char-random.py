from __future__ import print_function

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import optimizers
import multiprocessing
import os
import sys
import threading
import rdock_util
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import numpy as np
np.random.seed(0)

def preprocess(smiles, length=128):
    smiles = smiles.replace('Cl', '?')
    smiles = smiles.replace('Br', '*')
    smiles = '^'+smiles+'$'*(length-len(smiles)-1)
    return smiles

train_smiles = []
filename = '250k_rndm_zinc_drugs_clean.smi'
l = 0
with open(filename) as f:
    for line in f:
        smiles = line.rstrip()
        l = max(l, len(smiles))
        train_smiles.append(preprocess(smiles))

def build_vocab(smiles):
    i = 0
    char_dict, ord_dict = {}, {}
    for smile in smiles:
        for c in smile:
            if c not in char_dict:
                char_dict[c] = i
                ord_dict[i] = c
                i += 1
    return char_dict, ord_dict

char_dict, ord_dict = build_vocab(train_smiles)
vocab_size = len(char_dict)

def encode(smiles):
    smiles = preprocess(smiles)
    return [char_dict[c] for c in smiles]

def decode(sequence):
    smiles = ''.join([ord_dict[s] for s in sequence])
    smiles = smiles.replace('^', '')
    smiles = smiles.replace('$', '')
    smiles = smiles.replace('*', 'Br')
    smiles = smiles.replace('?', 'Cl')
    return smiles

class RNN(chainer.Chain):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256):
        super(RNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, emb_dim)
            self.lstm1 = L.LSTM(emb_dim, hidden_dim)
            self.out = L.Linear(hidden_dim, vocab_size)
    
    def reset_state(self):
        self.lstm1.reset_state()
    
    def forward(self, x):
        h = self.embed(x)
        h = self.lstm1(h)
        h = self.out(h)
        pred = F.softmax(h)
        return pred

    def __call__(self, x, t):
        h = self.embed(x)
        h = self.lstm1(h)
        h = self.out(h)
        return F.softmax_cross_entropy(h, t)

rnn = RNN(vocab_size)
optimizer = optimizers.Adam()
optimizer.setup(rnn)
model_name = "char-model-9.npz"
if os.path.exists(model_name):
    print("loading model")
    serializers.load_npz(model_name, rnn)

#print("start pre-training")
#for _ in range(10000):
#    idx = np.random.choice(len(train_smiles))
#    smiles = train_smiles[idx]
#    sequence = np.array([encode(smiles)]).astype(np.int32)
#    loss = 0
#    rnn.reset_state()
#    for t in range(len(sequence[0])-1):
#        with chainer.using_config('train', True):
#             loss += rnn(sequence[:, t], sequence[:, t+1])
#             if t % 32 == 31:
#                 rnn.cleargrads()
#                 loss.backward()
#                 loss.unchain_backward()
#                 optimizer.update()
#    serializers.save_npz(model_name, rnn)
#print("finish pre-training")

elapsed_min = 0
def current_best(scores):
    global elapsed_min
    elapsed_min += 1
    valid = 0
    good = 0
    very_good = 0
    best_score = 1e10
    if scores != []:
        best_score = min(scores)
        total = len(scores)
        valid = len([s for s in scores if s < 1e10])
        good = len([s for s in scores if s < -20])
        unique_good = len([s for s in list(set(scores)) if s < -20])
        very_good = len([s for s in scores if s < -30])
        unique_very_good = len([s for s in list(set(scores)) if s < -30])
    print("{},{},{},{},{},{},{},{}".format(elapsed_min, best_score, total, valid, good, unique_good, very_good, unique_very_good))
    t = threading.Timer(60, current_best, [scores])
    t.start()

valid_molecules = []
scores = []
trial = 100000
t = threading.Timer(60, current_best, [scores])
t.start()
for i in range(trial):
    smiles_list = []
    for _ in range(16):
        s = char_dict['^']
        sample_seq = [s]
        rnn.reset_state()
        for t in range(128):
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    prob = rnn.forward(np.array([s]).astype(np.int32)).data[0]
                    s = np.random.choice(vocab_size, p=prob)
                    sample_seq.append(s)
        smiles = decode(sample_seq)
        smiles_list.append(smiles)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    scores_t = pool.map(rdock_util.score, smiles_list)
    pool.close()
    pool.terminate()
    scores.extend(scores_t)
    for i in range(len(scores_t)):
        if scores_t[i] < 1e10:
            print(smiles_list[i], file=sys.stderr)
        valid_molecules.append((scores_t[i], smiles_list[i]))
valid_molecules.sort()
for score, smiles in valid_molecules:
    print(score, smiles)
