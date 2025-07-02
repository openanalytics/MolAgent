from rdkit import Chem
import numpy as np
import torch
import hashlib
import os.path as osp
import pickle
import shutil
from rdkit.Chem.AllChem import Get3DDistanceMatrix
from rdkit.Chem.rdmolops import GetDistanceMatrix
import pandas as pd
#from torch_geometric.data import Data, InMemoryDataset,download_url
from tqdm import tqdm
from automol.structurefeatures.GradFormer.utils_3d.rdkit_util import cal_atomwise_Delta_sasa ,set_atomwise_TPSA, cal_sasa ,get_lone_pairs
#### https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
# allowable multiple choice node and edge features 
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_number_lone_pairs_list': [0, 1, 2, 3, 4, 'misc'],
    
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1
# # miscellaneous case
# i = safe_index(allowable_features['possible_atomic_num_list'], 'asdf')
# assert allowable_features['possible_atomic_num_list'][i] == 'misc'
# # normal case
# i = safe_index(allowable_features['possible_atomic_num_list'], 2)
# assert allowable_features['possible_atomic_num_list'][i] == 2

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            safe_index(allowable_features['possible_number_lone_pairs_list'],get_lone_pairs(atom) ),
        ]
    return atom_feature
# from rdkit import Chem
# mol = Chem.MolFromSmiles('Cl[C@H](/C=C/C)Br')
# atom = mol.GetAtomWithIdx(1)  # chiral carbon
# atom_feature = atom_to_feature_vector(atom)
# assert atom_feature == [5, 2, 4, 5, 1, 0, 2, 0, 0]


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list'],
        allowable_features['possible_number_lone_pairs_list']
        ]))

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature
# uses same molecule as atom_to_feature_vector test
# bond = mol.GetBondWithIdx(2)  # double bond with stereochem
# bond_feature = bond_to_feature_vector(bond)
# assert bond_feature == [1, 2, 0]

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))

def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx, 
    chirality_idx,
    degree_idx,
    formal_charge_idx,
    num_h_idx,
    number_radical_e_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict
# # uses same atom_feature as atom_to_feature_vector test
# atom_feature_dict = atom_feature_vector_to_dict(atom_feature)
# assert atom_feature_dict['atomic_num'] == 6
# assert atom_feature_dict['chirality'] == 'CHI_TETRAHEDRAL_CCW'
# assert atom_feature_dict['degree'] == 4
# assert atom_feature_dict['formal_charge'] == 0
# assert atom_feature_dict['num_h'] == 1
# assert atom_feature_dict['num_rad_e'] == 0
# assert atom_feature_dict['hybridization'] == 'SP3'
# assert atom_feature_dict['is_aromatic'] == False
# assert atom_feature_dict['is_in_ring'] == False

def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx, 
    bond_stereo_idx,
    is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict

################################################################################
####### after  https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py


#if abs(dsa) >0 : 
#print(atom.GetSymbol(), atom.GetProp('SASAClassName'), atom.GetDoubleProp("SASA"),  atom.GetDoubleProp("free_SASA"))




def mol2graph(mol , add_sasa=True, add_dist_mtx=True):
    """
    Converts rdkit to graph Data object
    :input: rdkit mol
    :return: graph object
    """
    graph = dict()
    # atoms
    atom_features_list = []
    sasa_atom_wise_list=[]
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    if add_sasa:
        if not  mol.GetAtoms()[0].HasProp("SASA"): cal_sasa(mol, total_only=True)
        if not  mol.GetAtoms()[0].HasProp("TPSA"): set_atomwise_TPSA(mol)
        for atom in mol.GetAtoms():
            aa=[0.,0.,0.,0.]#{"SASA":0.0,'Polar':0.0, 'Apolar':0.0, 'TPSA':0.0} 
            aa[0] = atom.GetDoubleProp("SASA")
            if atom.GetProp('SASAClassName')=='Polar': aa[1] = atom.GetDoubleProp("SASA")
            else: aa[2] = atom.GetDoubleProp("SASA") 
            aa[3] = atom.GetDoubleProp("TPSA") 
            sasa_atom_wise_list.append(aa)
        graph['sasa_atom_wise'] =np.array( sasa_atom_wise_list, dtype = np.float64)
            
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    if add_dist_mtx:
        graph["dist_mat"]=Get3DDistanceMatrix(mol)
        graph["topological_dist_mat"]=GetDistanceMatrix(mol)
    
    return graph 

#######################################
def sdf2graph(sdf_file, removeHs=False):
    """
    Converts sdf to graph Data object
   
    """
    mol= Chem.SDMolSupplier(sdf_file,removeHs=removeHs)[0]
    #mol = Molecule.from_rdkit(mol)
    #mol = mol if removeHs else Chem.AddHs(mol)
    graph=mol2graph(mol)
    return graph
##############################################
def ReorderCanonicalRankAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order




#######################################
def smiles2graph(smiles_string, removeHs=False, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    #graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    """

    mol = Chem.MolFromSmiles(smiles_string)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)
    graph=mol2graph(mol)
    return graph


