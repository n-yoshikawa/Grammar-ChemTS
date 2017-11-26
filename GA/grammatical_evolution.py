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
import copy

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

def CFGtoGene(prod_rules, max_len=-1):
    gene = []
    for r in prod_rules:
        lhs = GCFG.productions()[r].lhs()
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions()) if rule.lhs() == lhs]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [0]*(max_len-len(gene))
    return gene

def GenetoCFG(gene):
    prod_rules = []
    stack = [GCFG.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except:
            break
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions()) if rule.lhs() == lhs]
        rule = possible_rules[g%len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal) 
                                and (str(a) != 'None'),
                                zinc_grammar.GCFG.productions()[rule].rhs())
        stack.extend(list(rhs)[::-1])
    return prod_rules


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
    population = []
    rules = np.load("rules.npz")['arr_0']
    initial_rules = np.random.choice(rules, 100)
    initial_genes = [CFGtoGene(rule, max_len=288) for rule in initial_rules]
    initial_scores = []
    for i in range(0, len(initial_genes), multiprocessing.cpu_count()):
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        initial_scores.extend(pool.map(rdock_util.score, [cfg_util.decode(GenetoCFG(gene))[0] for gene in initial_genes[i:i+multiprocessing.cpu_count()]]))
        pool.close()
        pool.terminate()
    for s, m in zip(initial_scores, initial_genes):
        population.append((s, m))

    trial = 0
    valid_smiles = []
    scores = []
    all_smiles = []
    t = threading.Timer(60, current_best, [scores])
    t.start()
    for generation in range(100):
        print("generation", generation)
        population = sorted(population, key=lambda x:x[0])[:100]
        for s, g in population:
            print(s, cfg_util.decode(GenetoCFG(g))[0])
        cpu_count = multiprocessing.cpu_count()
        # crossover
        children_smiles = []
        children_genes = []
        while len(children_smiles) < cpu_count*0.8:
            idx1, idx2 = np.random.choice(len(population), size=2)
            score1, gene1 = population[idx1]
            score2, gene2 = population[idx2]
            cut_point = np.random.choice(len(gene1))
            gene_child = gene1[:cut_point] + gene2[cut_point:]
            smiles_child = cfg_util.decode(GenetoCFG(gene_child))[0]
            if is_valid_smiles(smiles_child) and smiles_child not in all_smiles:
                children_smiles.append(smiles_child)
                children_genes.append(gene_child)
                all_smiles.append(smiles_child)
        # mutation
        while len(children_smiles) < cpu_count:
            idx = np.random.choice(len(population))
            score, gene = population[idx]
            mutation_idx = np.random.choice(len(gene))
            gene_mutant = copy.deepcopy(gene)
            gene_mutant[mutation_idx] = np.random.choice(80)
            smiles_mutant = cfg_util.decode(GenetoCFG(gene_mutant))[0]
            if is_valid_smiles(smiles_mutant) and smiles_mutant not in all_smiles:
                children_smiles.append(smiles_mutant)
                children_genes.append(gene_mutant)
                all_smiles.append(smiles_mutant)

        pool = multiprocessing.Pool(cpu_count)
        scores_child = pool.map(rdock_util.score, children_smiles)
        pool.close()
        pool.terminate()
        scores.extend(scores_child)
        assert(len(scores_child) == len(children_genes))
        for s, g in zip(scores_child, children_genes):
            if (s, g) not in population:
                population.append((s, g))

if __name__ == "__main__":
    main()
