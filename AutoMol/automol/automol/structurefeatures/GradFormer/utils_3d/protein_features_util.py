from rdkit import Chem
import numpy as np
import torch
import hashlib
import os.path as osp
import pickle
import shutil

import pandas as pd
from torch_geometric.data import Data, InMemoryDataset,download_url
from tqdm import tqdm
import numpy as np
from automol.structurefeatures.GradFormer.utils_3d.protein_data import total_AA_resisues_names
from automol.structurefeatures.GradFormer.utils_3d.get_interactions import protein_internal_coor
from automol.structurefeatures.GradFormer.utils_3d.residue import Molecule ,ResidueId , Residue

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

allowable_res_features = {
'possible_resname' : total_AA_resisues_names + ['UNK'],
    'possible_chirality_list' : [  'L', 'D', 'UNK' ],
    'possible_charged_Positive': [False, True],
    'possible_charged_Negative':[False, True],
    'possible_is_amide':[False, True],
    'possible_is_aromatic_list': [False, True],
    'possible_terminal_state': [ 'N',   'C', 'None'],
    'possible_covalent_bond_type_list' : ["peptide_bond", "disulfide_bond","other"]    
}
def residues_to_feature_vector(resname):
    """
    Converts rdkit residues object to feature list of indices
    :return: list
    """
    res_feature = [
            safe_index(allowable_res_features['possible_resname'], resname),
            safe_index(allowable_res_features['possible_chirality_list'], 'L'),
            allowable_res_features['possible_charged_Positive'].index(resname in  ['LYS','ARG']),
            allowable_res_features['possible_charged_Negative'].index(resname in  ['GLU','ASP']),
            allowable_res_features['possible_is_amide'].index(resname in  ['GLN','ASN']),
            allowable_res_features['possible_is_aromatic_list'].index(resname in  [ "HIS", "PHE","TRP", "TYR"]),
        safe_index(allowable_res_features['possible_terminal_state'], 'None')
            ]
    return res_feature
def get_res_feature_dims():
    return list(map(len, [
        allowable_res_features['possible_resname'],
        allowable_res_features['possible_chirality_list'],
        allowable_res_features[ 'possible_charged_Positive'],
        allowable_res_features[ 'possible_charged_Negative'],
        allowable_res_features['possible_is_amide'],
        allowable_res_features['possible_is_aromatic_list'],
        allowable_res_features['possible_terminal_state']
        ]))

############################# covalent bonds b. residues
allowable_res_re_bonds = {
    'possible_is_peptide_bond':[False, True],
    'possible_is_disulfide_bond': [False, True],
    'possible_is_UNK_bond': [False, True]     
}
def bond_to_feature_vector(bonds):
    """
    ["peptide_bond", "disulfide_bond","UNK"]

    """
    bond_feature = [
                allowable_res_re_bonds['possible_is_peptide_bond'].index( "peptide_bond" in bonds ),
                allowable_res_re_bonds['possible_is_disulfide_bond'].index("disulfide_bond" in bonds ),
                allowable_res_re_bonds['possible_is_UNK_bond'].index('UNK' in bonds)
            ]
    return bond_feature

def get_res_res_bond_feature_dims():
    return list(map(len, [
        allowable_res_re_bonds['possible_is_peptide_bond'],
        allowable_res_re_bonds['possible_is_disulfide_bond'],
        allowable_res_re_bonds['possible_is_UNK_bond']
        ]))
############################# Noncovalent bonds b. residues

allowable_Noncovalent_res_res_bonds = {
    'H-bonding':[False, True],
    'Halogen_bonding': [False, True],
    'PiStacking': [False, True]   ,  
    'Anionic-Cationic': [False, True]   , 
    'Cation-Pi': [False, True]   , 
     'Metal-Don-Accr': [False, True]   , 
    'VdWContact': [False, True]   , 
    'Hydrophobic': [False, True]  
}
def Noncovalent_bond_to_feature_vector(bonds):
    """

    """
    bond_feature = []
    for k ,v in allowable_Noncovalent_res_res_bonds.items() :
        bond_feature.append( v.index(k in bonds ))
    return bond_feature


def get_Noncovalent_bond_feature_dims():
    return len(allowable_Noncovalent_res_res_bonds)*[2]
import string
possible_chain_IDs=list(string.ascii_uppercase) +['UNK']
#safe_index(possible_chain_IDs, k)
################################
def protein_mol2graph(mol, add_rotamers=True,
                      add_res_res_bonds=True,
                      add_BB_dihs =True,
                          add_res_res_noncovalent_bonds=True,
                        add_sasa_residue_wise=True,
                      add_centriods_dist_mat=True
):
    """
    Converts rdkit to graph Data object
    :input: rdkit mol
    :return: graph object
    """
    if not add_res_res_bonds: add_BB_dihs=False
    get_internal_protein=protein_internal_coor(mol)
    if add_res_res_bonds:
        get_internal_protein.get_res_res_bonds()
        get_internal_protein.get_backbone_dihs()
    if add_rotamers: 
        get_internal_protein.get_rotamers()
    if add_res_res_noncovalent_bonds:
        get_internal_protein.get_res_res_noncovalent_bonds()
    if add_sasa_residue_wise:
        get_internal_protein.get_sasa_residue_wise()
    ID2resid_key={}
    resid_key2ID={}
    #get_internal_protein.res_res_bonds #        
    #get_internal_protein.backbone_dihs #   
    #get_internal_protein.rotamers
    #get_internal_protein.res_res_noncovalent_bonds
    graph = dict()
    ######################### residues info
    residues_features_list = []
    residues_rotamers_list = []
    sasa_residue_wise_list=[]
    chains=[]
    residue_ids=[]
    i=0
    for residue in mol:
        k=residue.resid
        if k.name=='':k.name='UNK'
        if k.chain=='': k.chain=None
        residue_ids.append(k.number)
        chains.append(safe_index(possible_chain_IDs, k.chain))
        ID2resid_key[i]  =  k
        resid_key2ID[k]=i
        residues_features_list.append(residues_to_feature_vector(k.name))
        ############## add rotamers
        chis=[np.nan,np.nan,np.nan,np.nan]
        if add_rotamers and k in get_internal_protein.rotamers:
            chidict=get_internal_protein.rotamers[k]
            if 'CHI1' in chidict: chis[0]=chidict['CHI1']
            if 'CHI2' in chidict: chis[1]=chidict['CHI2']
            if 'CHI3' in chidict: chis[2]=chidict['CHI3']
            if 'CHI4' in chidict: chis[3]=chidict['CHI4']      
        residues_rotamers_list.append(chis)
        ########## 
        #################################### add sasa
        if add_sasa_residue_wise and k in get_internal_protein.sasa_residue_wise:
            sasatmp= get_internal_protein.sasa_residue_wise[k]
            sasa_residue_wise_list.append([sasatmp['SASA'], sasatmp['Polar'], sasatmp['Apolar'], sasatmp['TPSA'] ])
        i+=1
    min_resid=min(residue_ids)
    residue_ids=[resid-min_resid for resid in residue_ids]
    graph['node_feat'] = np.array(residues_features_list, dtype = np.int64)
    graph['num_nodes'] = len(residues_features_list)
    if add_rotamers: 
        graph['rotamers'] =np.array( residues_rotamers_list, dtype = np.float32)
    if add_sasa_residue_wise:
        graph['sasa_residue_wise'] =np.array( sasa_residue_wise_list, dtype = np.float32)
    ######################   covalent bonds
    
    num_bond_features = 3  # "is peptide_bond", "is disulfide_bond","is UNK"
    if  add_res_res_bonds and len(get_internal_protein.res_res_bonds) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        BB_dihes_list=[]
        for k in get_internal_protein.res_res_bonds:
            if k[0] not in resid_key2ID:
                continue
            if k[1] not in resid_key2ID:
                continue
            ii = resid_key2ID[k[0]]
            j = resid_key2ID[k[1]]
            bonds=[ b for b in get_internal_protein.res_res_bonds[k]]
            edge_feature = bond_to_feature_vector(bonds)
            # add edges in both directions
            edges_list.append((ii, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, ii))
            edge_features_list.append(edge_feature)
            phi_psi= [np.nan,np.nan]
            if add_BB_dihs and k in  get_internal_protein.backbone_dihs:
                phi=get_internal_protein.backbone_dihs[k]['phi']
                psi=get_internal_protein.backbone_dihs[k]['psi']
                phi_psi=[phi, psi]
            BB_dihes_list.append(phi_psi)
            BB_dihes_list.append(phi_psi)
            
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)
        backbone_dih_bonds_feature= np.array(BB_dihes_list, dtype = np.float32)
    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    if  add_BB_dihs: 
        graph['res_res_BB_dihs']=backbone_dih_bonds_feature
    ######################   Noncovalent bonds
    num_Noncovalent_bond_features = len(allowable_Noncovalent_res_res_bonds) 
    
    if add_res_res_noncovalent_bonds and len(get_internal_protein.res_res_noncovalent_bonds) > 0: # mol has bonds
        noncovalent_edges_list = []
        noncovalent_edge_features_list = []
        for k in get_internal_protein.res_res_noncovalent_bonds:
            i = resid_key2ID[k[0]]
            j = resid_key2ID[k[1]]
            bonds=[ b for b in get_internal_protein.res_res_noncovalent_bonds[k]]
            edge_feature = Noncovalent_bond_to_feature_vector(bonds)
            # add edges in both directions
            noncovalent_edges_list.append((i, j))
            noncovalent_edge_features_list.append(edge_feature)
            noncovalent_edges_list.append((j, i))
            noncovalent_edge_features_list.append(edge_feature)

            
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        noncovalent_edge_index = np.array(noncovalent_edges_list, dtype = np.int64).T
        noncovalent_edge_attr = np.array(noncovalent_edge_features_list, dtype = np.int64)
        graph['noncovalent_edge_index'] = noncovalent_edge_index
        graph['noncovalent_edge_feat'] = noncovalent_edge_attr
    ##########################################
    if add_centriods_dist_mat:
        #dist_mat=get_internal_protein.get_3ddist_matrix( seletion='CA')
        centriods_dists=get_internal_protein.get_centriods_dists()
        centriods_dists_edges_index = []
        centriods_dists_values = []
        for k in centriods_dists:
            i = resid_key2ID[k[0]]
            j = resid_key2ID[k[1]]
            centriods_dists_edges_index.append((i, j))
            centriods_dists_values.append(centriods_dists[k])
        graph['centriods_dists_edges_index'] =  np.array(centriods_dists_edges_index, dtype = np.int64).T
        graph['centriods_dists'] = np.array( centriods_dists_values, dtype = np.float32)
        

    graph['nodeID2resinf']=ID2resid_key
    graph['residue_ids']= np.array(residue_ids, dtype = np.int64)
    graph['chain_ids']=    np.array(chains, dtype = np.int64) 
    return graph 

#################
def protein_pdb2graph(pdb_file, add_rotamers=True,
                      add_res_res_bonds=True,
                      add_BB_dihs =True,
                          add_res_res_noncovalent_bonds=True,
                        add_sasa_residue_wise=True
):
    rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    protein_mol = Molecule.from_rdkit(rdkit_prot)
    return  protein_mol2graph(protein_mol, add_rotamers=add_rotamers,
                      add_res_res_bonds=add_res_res_bonds,
                      add_BB_dihs =add_BB_dihs,
                          add_res_res_noncovalent_bonds=add_res_res_noncovalent_bonds,
                        add_sasa_residue_wise=add_sasa_residue_wise)


