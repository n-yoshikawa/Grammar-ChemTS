# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

import nltk
import copy
import smiles_util
import rdock_util
import multiprocessing
import threading

from math import *
import numpy as np

import os
import sys

np.random.seed(0)
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

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
print(l)

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

class State:
    def __init__(self, moves=None, rnn=None):
        self.rnn = copy.deepcopy(rnn)
        if moves is not None:
            self.moves = copy.deepcopy(moves)
            for m in self.moves:
                self.rnn.forward(m)
        else:
            self.moves = []

    def Clone(self):
        # Create a deep clone of this state.
        state = State(self.moves, self.rnn)
        return state

    def DoMove(self, move):
        # Update a state by carrying out the given move.
        self.moves.append(move)

    def Rollout(self):
        self.rnn.reset_state()
        for m in self.moves[:-1]:
            self.rnn.forward(np.array([m]).astype(np.int32))
        beam_width = 16 # must be bigger than 3!
        eps = 1e-100
        initial_char = self.moves
        candidates = [(self.rnn, initial_char, 0.0)]
        sequence_length = 250
        for t in range(sequence_length):
            next_candidates = []
            for previous_model, chars, log_likelihood in candidates:
                model = previous_model.copy()
                x = np.asarray([chars[-1]]).astype(np.int32)
                with chainer.using_config('train', False):
                    with chainer.no_backprop_mode():
                        probability = model.forward(x).data[0]
                log_probability = np.log(probability)
                order = probability.argsort()[:-beam_width:-1]
                for sampled_char in order:
                    if log_probability[sampled_char] > np.log(eps) + eps:
                        next_candidates.append(
                            (model, chars + [sampled_char],
                             log_likelihood + log_probability[sampled_char]))
            candidates = sorted(next_candidates, key=lambda x: -x[2])[:beam_width]
            if all([len(candidate[1]) == 0 for candidate in candidates]):
                break
        smiles = []
        self.moves_rollout = []
        for candidate in candidates:
            self.moves_rollout.append(candidate[1])
            smiles.append(decode(candidate[1]))
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        scores = pool.map(rdock_util.score, smiles)
        pool.close()
        pool.terminate()
        self.rollout_scores = scores
        self.rollout_smiles = smiles
        return [(score, smiles) for score, smiles in zip(scores, smiles)]
        
    def GetMoves(self):
        # Get all possible moves from this state.
        return [i for i in range(vocab_size)]
    
    def GetResult(self):
        # Get the result
        if self.rollout_scores == []:
            return (0.0, '')
        best = np.argmin(self.rollout_scores)
        s = self.rollout_scores[best]
        score = -np.tanh((s+20)/20)
        return (score, self.rollout_smiles[best])

    def __repr__(self):
        return "moves: {}, stack: {}".format(self.moves, self.stack)

class Node:
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the moves that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.sumScore = 0
        self.generatedSmiles = []
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes

        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        C = 2.0
        s = sorted(self.childNodes, key = lambda c: c.sumScore/c.visits + C*sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, state):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = state)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit, add score and record of result strings.
            result is a tuple of (score, smiles)
        """
        score = result[0]
        smiles = result[1]
        self.visits += 1
        self.sumScore += score
        if score > 0:
            self.generatedSmiles.append(smiles)

    def __repr__(self):
        return "move:" + str(self.move) + ", sumScore:" + str(self.sumScore) + ", visits:" + str(self.visits) + ", untriedMoves:" + str(self.untriedMoves)

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1, indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s

elapsed_min = 0
def current_best(scores):
    global elapsed_min
    elapsed_min += 1
    valid = 0
    good = 0
    very_good = 0
    best_score = 1e10
    total = 0
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
    
def MCTS(rootstate, itermax=100, verbose=False):
    rootnode = Node(state = rootstate)
    results = []
    scores = []
    t = threading.Timer(60, current_best, [scores])
    t.start()

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        if verbose: print("start select")
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            if verbose: print("move:", node.move)
            state.DoMove(node.move)
        if verbose: print("finished select")

        # Expand
        if verbose: print("start expand")
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = np.random.choice(node.untriedMoves) 
            if verbose: print("move:", m)
            state.DoMove(m)
            node = node.AddChild(m, state) # add child and descend tree
        if verbose: print("finish expand")

        # Rollout
        if verbose: print("start rollout")
        result = state.Rollout()
        for score, smiles in result:
            scores.append(score)
            if score < 1e10:
                print(score, smiles, file=sys.stderr)
        if verbose: print("finish rollout")

        # Backpropagate
        if verbose: print("start backpropagate")
        result = state.GetResult()
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(result)
            node = node.parentNode
        if verbose: print("finish backpropagate")
    return rootnode.generatedSmiles

def main():
    rnn = RNN(vocab_size)
    if os.path.exists("char-model-9.npz"):
        serializers.load_npz("char-model-9.npz", rnn)
        print("loaded model successfully")
    optimizer = optimizers.Adam()
    optimizer.setup(rnn)

    rootstate = State(rnn=rnn)
    smiles = MCTS(rootstate, 10000)

    #for _ in range(100):
    #    serializers.save_npz("model.npz", rnn)
    #    print("model saved.")
    #    print("finish pre-training")
    #    rootstate = State(rnn=rnn)
    #    smiles = MCTS(rootstate, 10000)

    #    print("start pre-training")
    #    for epoch in range(10000):
    #        sequence = np.array([np.random.choice(train_rules)]).astype(np.int32)
    #        loss = 0
    #        rnn.reset_state()
    #        for t in range(len(sequence[0])-1):
    #            with chainer.using_config('train', True):
    #                 loss += rnn(sequence[:, t], sequence[:, t+1])
    #                 if t % 32 == 0 or t==len(sequence[0])-2:
    #                     rnn.cleargrads()
    #                     loss.backward()
    #                     loss.unchain_backward()
    #                     optimizer.update()
if __name__ == "__main__":
    main()
