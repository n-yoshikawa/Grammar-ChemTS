from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import multiprocessing
from datetime import datetime

def rdock_score(smiles, num_docking=3):
    print("start!:", smiles)
    docking_result_file = '{}_out'.format(smiles)
    sdf_name = '{}.sdf'.format(smiles)
    score_name = '<SCORE>' # <SCORE> or <SCORE.INTER>

    min_score = 1e10

    # Translation from SMILES to sdf
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
            cmd = '$RBT_ROOT/build/exe/rbdock -r cavity.prm -p $RBT_ROOT/data/scripts/dock.prm -i {} -o {} -T 1 -n {} >/dev/null'.format(sdf_name, docking_result_file, num_docking)
            proc = subprocess.call(cmd, shell=True)

            #----find the minimum score of rdock from multiple docking results
            f = open(docking_result_file+'.sd')
            lines = f.readlines()
            f.close()

            isScore = False
            for line in lines:
                if isScore:
                    min_score = min(float(line), min_score)
                    isScore = False
                if score_name in line: # next line has score
                    isScore = True

            print(smiles, 'minimum rdock score', min_score)
    except:
        import traceback
        traceback.print_exc()
    return min_score

print("start:", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
smiles_list = ['c1ccccc1', 'Cc1ccccc1', 'Oc1ccccc1']
pool = multiprocessing.Pool(multiprocessing.cpu_count())
scores = pool.map(rdock_score, smiles_list)
print(scores)
print("end:", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
