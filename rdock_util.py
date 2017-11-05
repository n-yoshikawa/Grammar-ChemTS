from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import multiprocessing
from datetime import datetime
import hashlib
import traceback
import os


def score(smiles, num_docking=1):
    smiles_md5 = str(hashlib.md5(smiles.encode('utf-8')).hexdigest())
    docking_result_file = '{}_out'.format(smiles_md5)
    sdf_name = '{}.sdf'.format(smiles_md5)
    score_name = '<SCORE>' # <SCORE> or <SCORE.INTER>

    min_score = 1e10

    # Translation from SMILES to sdf
    if smiles == '':
        mol = None
    else:
        mol = Chem.MolFromSmiles(smiles)
    try:
        if mol is not None:
            mol = Chem.AddHs(mol)
            cid = AllChem.EmbedMolecule(mol)
            opt = AllChem.UFFOptimizeMolecule(mol,maxIters=200)
            fw = Chem.SDWriter(sdf_name)
            fw.write(mol)
            fw.close()

            #----rdock calculation
            cmd = '$RBT_ROOT/build/exe/rbdock -r cavity.prm -p $RBT_ROOT/data/scripts/dock.prm -i {} -o {} -T 1 -n {}'.format(sdf_name, docking_result_file, num_docking)
            path = docking_result_file+'.sd'
            if not os.path.exists(path):
                FNULL = open(os.devnull, 'w')
                proc = subprocess.call(cmd, shell=True, stdout=FNULL)

            #----find the minimum score of rdock from multiple docking results
            if os.path.exists(path):
                with open(path, 'r') as f:
                    lines = f.readlines()
                isScore = False
                for line in lines:
                    if isScore:
                        min_score = min(float(line), min_score)
                        isScore = False
                    if score_name in line: # next line has score
                        isScore = True
    except:
        traceback.print_exc()
    return min_score
