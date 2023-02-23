import py3Dmol
import pandas as pd
from stmol import showmol
# importing from rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

def makeblock(smi):
    if valid_smiles(smi):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol)
            mblock = Chem.MolToMolBlock(mol)
        except ValueError:
            return None
        return mblock

def render_mol(xyz,style):
    xyzview = py3Dmol.view(width=665,height=400)#(width=400,height=400)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({style.lower():{}})
    xyzview.setBackgroundColor('#0E1117')   #16181E
    xyzview.zoomTo()
    showmol(xyzview,height=400,width=665)
    
def render_prot(xyz,style,spin, width=665, height=400):
    
    view = py3Dmol.view(query=f'pdb:{xyz}', width=width, height=height)
    view.setStyle({style.lower():{'color':'spectrum'}})
    view.setBackgroundColor('#0E1117')
    view.spin(spin)
    view.zoomTo()
    showmol(view, width=width, height=height)
    
def computeFP(smiles, labels=None):
    moldata = [Chem.MolFromSmiles(mol) for mol in smiles]
    fpdata=[]
    for i, mol in enumerate(moldata):
        if mol:
            ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
            fpdata += [ecfp6]
    fp_df = pd.DataFrame(data=fpdata, index=smiles)
    if labels is not None: fp_df['labels'] =labels
    return fp_df


def valid_smiles(smi):
    if Chem.MolFromSmiles(smi) is not None:
        return True
    else:
        return False

    