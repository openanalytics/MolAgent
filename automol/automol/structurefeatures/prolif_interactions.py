from math import log10,log
import numpy as np, pandas as pd
import os 
from rdkit import Chem
import prolif as plf
from rdkit.Chem import DataStructs

from automol.feature_generators import FeatureGenerator

class ProlifInteractionCountGenerator(FeatureGenerator):
    """
    feature generator returning the interaction features
    """

    def __init__(self,pdb_folder='',
                    sdf_file='',
                     name='prolif',
                     properties= ['pIC50'] ,
                     interactions=['Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic', 'Cationic', 'CationPi', 'PiCation', 'VdWContact']
                ):
        """
        Initialization 
        """
        self.properties=properties
        self.interactions=interactions
        self._setup_data(pdb_folder,sdf_file)
        ## list of names of the features
        self.names= [f'automol_prolif_feature_{name}_{i}' for i in range(len(interactions))]
        ## number of features
        self.nb_features=len(interactions)
        ## generator name
        self.generator_name=f'automol_prolif_feature_{name}'
                
    def _setup_data(self,pdb_folder,sdf_file):
        
        subd=next(os.walk(pdb_folder))[2]
        self.pdb_dict={}
        for pdb_idx,d in enumerate(subd):
            pdb_file=f'{pdb_folder}/{d}'
            pdb_id=d.rsplit('_',1)[0]
            rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False)
            if rdkit_prot is None:
                rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False,sanitize=False, proximityBonding=True)
                Chem.GetSymmSSSR(rdkit_prot)
            prolif_protein = plf.Molecule(rdkit_prot)
            self.pdb_dict[pdb_id]=prolif_protein
        
        self.ligandsmiles_to_complex={}
        for idx,mol in enumerate(plf.sdf_supplier(sdf_file)):
            from automol.standardize import standardize
            pdb_nm=mol.GetProp('pdb')
            prolif_ligand = plf.Molecule(mol)
            self.ligandsmiles_to_complex[standardize(Chem.MolToSmiles(mol))]={'pdb':pdb_nm, 'prolif_ligand': prolif_ligand}

    def _get_count_vector(self,df,row=0, interactions=['Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic', 'Cationic', 'CationPi', 'PiCation', 'VdWContact']):
        feat=np.zeros(len(interactions),dtype=int)
        for idx,inter in enumerate(interactions):
            try:
                feat[idx]=np.sum(df.xs(inter, level="interaction", axis=1).iloc[row,:])
            except:
                feat[idx]=0
        return feat    
    
    def generate(self,smiles):
        complexes=[(idx, self.ligandsmiles_to_complex[smi]) for idx,smi in enumerate(smiles)]

        merged_pdb_ids={}
        for idx,cplx in complexes:
            if cplx['pdb'] not in merged_pdb_ids:
                merged_pdb_ids[cplx['pdb']]=[]
            merged_pdb_ids[cplx['pdb']].append((idx,cplx['prolif_ligand']))

        X=np.zeros((len(smiles),self.nb_features))
        for key,item in merged_pdb_ids.items():
            fp = plf.Fingerprint(interactions=self.interactions,
                                 vicinity_cutoff=8.0,
                                 count=True)
            # run on your poses
            protein_mol=self.pdb_dict[key]
            fp.run_from_iterable([mol for idx,mol in item], protein_mol,n_jobs=1,progress=False)
            df=fp.to_dataframe()
            for row,(smiles_idx,mol) in enumerate(item):
                X[smiles_idx,:] = self._get_count_vector(df, row=row, interactions=self.interactions)

        return X
    
