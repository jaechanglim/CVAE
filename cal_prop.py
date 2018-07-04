from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumHBD
from rdkit.Chem.rdMolDescriptors import CalcNumHBA
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem
from multiprocessing import Pool
def cal_prop(s):
    m = Chem.MolFromSmiles(s)
    if m is None : return None
    return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcNumHBD(m), CalcNumHBA(m), CalcTPSA(m)

f = open('smiles.txt')
smiles = f.read().split('\n')[:-1]
pool = Pool(8)

r = pool.map_async(cal_prop, smiles)

data = r.get()
pool.close()
pool.join()
w = open('smiles_prop.txt', 'w')
for d in data:
    if d is None:
        continue
    w.write(d[0] + '\t' + str(d[1]) + '\t'+ str(d[2]) + '\t'+ str(d[3]) + '\t' + str(d[4]) + '\t' + str(d[5]) + '\n')
w.close()    

