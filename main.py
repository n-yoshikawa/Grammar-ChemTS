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

import nltk
import copy
import zinc_grammar
import cfg_util
import smiles_util

from math import *
import numpy as np

np.random.seed(0)

class State:
    def __init__(self, moves=None, stack=None):
        if stack is None:
            self.moves = []
            self.stack = [str(zinc_grammar.GCFG.productions()[0].lhs())]
        else:
            self.moves = moves
            self.stack = stack
    def Clone(self):
        # Create a deep clone of this state.
        state = State(copy.deepcopy(self.moves), copy.deepcopy(self.stack))
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
        size = 0
        while state.GetMoves() != []: # while state is non-terminal
            m = np.random.choice(state.GetMoves())
            if verbose: print("move:", m)
            state.DoMove(m)
            size += 1
            if size > 300:
                break
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
    rootstate = State()
    smiles = MCTS(rootstate, 100000)
    print(smiles)

if __name__ == "__main__":
    main()
