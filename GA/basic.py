import numpy as np
import nltk
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors
import threading
import rdock_util
import multiprocessing
import sys
import zinc_grammar
import cfg_util

rdBase.DisableLog('rdApp.error')
np.random.seed(0)

GCFG = zinc_grammar.GCFG
rule_num = len(GCFG.productions())

class Molecule:
    def __init__(self, rules):
        self.rules = list(rules)
        self.smiles = cfg_util.decode(rules)[0]
        self.subtreesize = [-1 for _ in range(len(self.rules))]
        self.__calc_subtreesize(0)

    def __calc_subtreesize(self, i):
        if self.subtreesize[i] != -1:
            return self.subtreesize[i]

        rhs = list(map(str, filter(lambda x: (type(x) == nltk.grammar.Nonterminal) 
                                and (str(x) != 'None'),
                                GCFG.productions()[self.rules[i]].rhs())))
        if len(rhs) == 0:
            self.subtreesize[i] = 1
            return 1
        else:
            size = 1
            for _ in range(len(rhs)):
                size += self.__calc_subtreesize(i+size)
            self.subtreesize[i] = size
            return size

def crossover(mol1, mol2):
    children = []
    rules1 = mol1.rules
    rules2 = mol2.rules 
    for idx1 in range(len(rules1)):
        root1 = str(GCFG.productions()[rules1[idx1]].lhs())
        subtreesize1 = mol1.subtreesize[idx1]
        for idx2 in range(len(rules2)):
            root2 = str(GCFG.productions()[rules2[idx2]].lhs())
            if root1 == root2:
                subtreesize2 = mol2.subtreesize[idx2]
                new_rule1 = rules1[:idx1] \
                           + rules2[idx2:idx2+subtreesize2] \
                           + rules1[idx1+subtreesize1:]
                new_rule2 = rules2[:idx2] \
                           + rules1[idx1:idx1+subtreesize1] \
                           + rules2[idx2+subtreesize2:]
                children.append(Molecule(new_rule1))
                children.append(Molecule(new_rule2))
    return children

from rdkit.Chem import AllChem as Chem
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return smiles != '' and mol is not None and Descriptors.MolWt(mol) < 500

elapsed_min = 0
def current_best(scores):
    global elapsed_min
    elapsed_min += 1
    best_score = 1e10
    total = 0
    valid = 0
    good = 0
    unique_good = 0
    very_good = 0
    unique_very_good = 0
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

def main():
    rules = np.load("rules.npz")['arr_0']
    initial_rules = np.random.choice(rules, 100)
    mols = []
    for r in initial_rules:
        mols.append(Molecule(r))
    scores = []
    for i in range(0, len(mols), multiprocessing.cpu_count()):
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        scores.extend(pool.map(rdock_util.score, [m.smiles for m in mols[i:i+multiprocessing.cpu_count()]]))
        pool.close()
        pool.terminate()
    population = []
    for s, m in zip(scores, mols):
        population.append((s, m))
    print("initial population:", population)

    trial = 0
    valid_smiles = []
    scores = []
    t = threading.Timer(60, current_best, [scores])
    t.start()
    for generation in range(100):
        print("generation", generation)
        population = sorted(population, key=lambda x:x[0])[:100]
        for s, m in population:
            print(s, m.smiles)
        idx1, idx2 = np.random.choice(len(population), size=2)
        score1, mol1 = population[idx1]
        score2, mol2 = population[idx2]
        children = crossover(mol1, mol2)
        valid_children_smiles = []
        valid_children = []
        for child in children:
            if is_valid_smiles(child.smiles):
                valid_children.append(child)
                valid_children_smiles.append(child.smiles)
        unique_valid_children = []
        unique_valid_children_smiles = []
        for i in range(len(valid_children_smiles)):
            if valid_children_smiles[i] not in unique_valid_children_smiles:
                unique_valid_children_smiles.append(valid_children_smiles[i])
                unique_valid_children.append(valid_children[i])
        unique_valid_children_scores = []
        for i in range(0, len(unique_valid_children_smiles), multiprocessing.cpu_count()):
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            scores_chunk = pool.map(rdock_util.score, unique_valid_children_smiles[i:i+multiprocessing.cpu_count()])
            pool.close()
            pool.terminate()
            unique_valid_children_scores.extend(scores_chunk)
            scores.extend(scores_chunk)
        assert(len(unique_valid_children_scores) == len(unique_valid_children))
        for s, m in zip(unique_valid_children_scores, unique_valid_children):
            population.append((s, m))
if __name__ == "__main__":
    main()
