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
import zinc_grammar
import cfg_util
import smiles_util

from math import *
import numpy as np

np.random.seed(0)

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
        pred = F.softmax(h)
        return pred

    def __call__(self, x, t):
        h = self.embed(x)
        h = self.lstm1(h)
        h = self.out(h)
        return F.softmax_cross_entropy(h, t)

class State:
    def __init__(self, moves=None, stack=None, rnn=None):
        self.rnn = copy.deepcopy(rnn)
        self.rnn.reset_state()
        if stack is None:
            self.moves = []
            self.stack = [str(zinc_grammar.GCFG.productions()[0].lhs())]
        else:
            self.moves = copy.deepcopy(moves)
            self.stack = copy.deepcopy(stack)
            for s in self.moves:
                self.rnn.forward(s)

    def Clone(self):
        # Create a deep clone of this state.
        state = State(self.moves, self.stack, self.rnn)
        return state

    def DoMove(self, move):
        # Update a state by carrying out the given move.
        a = self.stack.pop()
        if a != str(zinc_grammar.GCFG.productions()[move].lhs()):
            raise ValueError("Error: rule {} can't be applied in this state.".format(move))

        rhs = filter(lambda x: (type(x) == nltk.grammar.Nonterminal) 
                                and (str(x) != 'None'),
                                zinc_grammar.GCFG.productions()[move].rhs())
        self.stack.extend(list(map(str, rhs))[::-1])
        self.moves.append(move)
        self.rnn.forward(np.array([move]).astype(np.int32))

    def Rollout(self):
        eps = 1e-100
        lhs_list = zinc_grammar.lhs_list
        lhs_map = zinc_grammar.lhs_map
        S = self.stack
        sampled_rules = []
        rule = np.array([self.moves[-1]]).astype(np.int32)
        sequence_length = 300
        for t in range(sequence_length):
            try:
                a = S.pop()
            except:
                break
            next_nonterminal = lhs_map[a]
            with chainer.using_config('train', False):
                with chainer.no_backprop_mode():
                    unmasked_probability = self.rnn.forward(rule).data[0]
            mask = zinc_grammar.masks[next_nonterminal]
            # Gumbel-max trick: 
            masked_probability = np.log(np.exp(unmasked_probability) * mask + eps)
            rule = np.argmax(masked_probability + np.random.gumbel(size=len(zinc_grammar.GCFG.productions())))
            sampled_rules.append(rule)
            rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                          zinc_grammar.GCFG.productions()[rule].rhs())
            S.extend(list(map(str, rhs))[::-1])
            rule = np.array([rule]).astype(np.int32)
        self.moves = self.moves + sampled_rules

        
    def GetMoves(self):
        # Get all possible moves from this state.
        if self.stack == []:
            return []
        else:
            return [rule for rule, prod in enumerate(zinc_grammar.GCFG.productions()) \
                         if str(prod.lhs()) == self.stack[-1]]
    
    def GetResult(self):
        # Get the result
        smiles = cfg_util.decode(self.moves)
        if smiles_util.verify_sequence(smiles):
            score = 1.0
        else:
            score = 0.0
        return (score, smiles)

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
        s = sorted(self.childNodes, key = lambda c: c.sumScore/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
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

def MCTS(rootstate, itermax=100, verbose=False):
    rootnode = Node(state = rootstate)

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
        state.Rollout()
        if verbose: print("finish rollout")

        # Backpropagate
        if verbose: print("start backpropagate")
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult())
            node = node.parentNode
        if verbose: print("finish backpropagate")

        print(state.GetResult())
    return rootnode.generatedSmiles

def main():
    print("loading data")
    train_smiles = []
    filename = '250k_rndm_zinc_drugs_clean.smi'
    with open(filename) as f:
        for line in f:
            smiles = line.rstrip()
            train_smiles.append(smiles)
            if len(train_smiles) > 100:
                break
    print("converting data")
    train_rules = cfg_util.encode(train_smiles)
    print("finished converting data")

    rnn = RNN(rule_size=len(zinc_grammar.GCFG.productions()))
    serializers.load_npz("model.npz", rnn)
    optimizer = optimizers.Adam()
    optimizer.setup(rnn)

    #print("start pre-training")
    #for epoch in range(1000):
    #    sequence = np.array([np.random.choice(train_rules)]).astype(np.int32)
    #    loss = 0
    #    rnn.reset_state()
    #    for t in range(len(sequence[0])-1):
    #        with chainer.using_config('train', True):
    #             loss += rnn(sequence[:, t], sequence[:, t+1])
    #             if t % 32 == 0 or t==len(sequence[0])-2:
    #                 rnn.cleargrads()
    #                 loss.backward()
    #                 loss.unchain_backward()
    #                 optimizer.update()
    #    if epoch % 100 == 0:
    #        serializers.save_npz("model.npz", rnn)
    #        print("model saved.")
    #print("finish pre-training")

    rootstate = State(rnn=rnn)
    smiles = MCTS(rootstate, 100000)
    print(smiles)

if __name__ == "__main__":
    main()
