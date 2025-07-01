"""RDKIT standardization

Authors: Joris Tavernier and Marvin Steijaert

Contact: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

All rights reserved, Open Analytics NV, 2021-2025. 
"""

from rdkit.Chem import MolFromSmiles, MolToSmiles, RemoveHs, MolFromSmarts 
from rdkit import rdBase;rdBase.DisableLog('rdApp.*')

from typing import Optional


def standardize(smiles: Optional[str]) -> Optional[str]:
    def _MolWithoutIsotopesToSmiles(mol):
        atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
        for atom, isotope in atom_data:
        # restore original isotope values
            if isotope:
                atom.SetIsotope(0)
        RemoveHs(mol)
        smiles = MolToSmiles(mol, canonical = True)
        m=MolFromSmiles(smiles)
        if m is None: return None
        return MolToSmiles(m, canonical = True)
    def neutralize_atoms(mol):
        pattern = MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4]),-1!$([*]~[1+,2+,3+,4+])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol
    if smiles is None:
             return None
    assert isinstance(smiles, str)  # if it is not a string, something strange happened
    smiles = smiles.split(' ')[0]
    if len(smiles)==0:  # empty strings are as good as None's
        return None
    try:
        m=MolFromSmiles(smiles)
    except:
        return None
    if m is None:
        return None
    
    try:
        m=neutralize_atoms(m)
    except:
        return None

    #RemoveStereochemistry(m)
    if m is None:
        return None
    return _MolWithoutIsotopesToSmiles(m)

__all__ = ['standardize']