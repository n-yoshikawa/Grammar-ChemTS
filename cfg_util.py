import nltk
import zinc_grammar

import numpy as np

def get_zinc_tokenizer(cfg):
    long_tokens = [a for a in cfg._lexical_index.keys() if len(a) > 1]
    replacements = ['$','%','^']
    assert len(long_tokens) == len(replacements)
    for token in replacements: 
        assert token not in cfg._lexical_index
    
    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens
    return tokenize

def encode(smiles):
    assert type(smiles) == list
    GCFG = zinc_grammar.GCFG
    tokenize = get_zinc_tokenizer(GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(GCFG)
    parse_trees = [parser.parse(t).__next__() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    productions = GCFG.productions()
    prod_map = {}
    for ix, prod in enumerate(productions):
        prod_map[prod] = ix
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    return indices

def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''

def decode(rule):
    productions = zinc_grammar.GCFG.productions()
    prod_seq = [productions[i] for i in rule]
    return prods_to_eq(prod_seq)
