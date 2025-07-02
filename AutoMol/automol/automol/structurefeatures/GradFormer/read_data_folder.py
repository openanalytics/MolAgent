import os
from  utils_3d.protein_features_util import protein_pdb2graph 
from  dataset_utils.dataset_3d import ResidueGranularityDataset
import pandas as pd
import torch_geometric.transforms as T
from  utils_3d.ligand_features_util import sdf2graph
from dataset_utils.dataset_3d import atomistic3d_InMemoryDataset_SDFfile
from  dataset_utils.dataset_3d import InteractionDataset_SDFfile
import os.path as osp
import torch
from rdkit import Chem
from dataset_utils.dataset_3d import CumulativeDataset, alldata_collate_fn,fn_sparse2dense, fn_interaction_pad, collate_tensor_fn, generate_fixed_coor
from dataset_utils.dataset_3d import CumulativePairedDataset, generate_hierarchical_triplets,update_fixed_coor, retrieve_prop_from_mol, update_fixed_coor
from sklearn.model_selection import train_test_split

def split_pairs_on_identifier(paired_test_triplets,split_number,max_ligand_pairs):
    both_below_pairs=[]
    split_pairs=[]
    both_above_pairs=[]
    for p,v in paired_test_triplets:
        for i,j in v:
            if i <= split_number and j <= split_number:
                both_below_pairs.append((p,[(i,j)]))
            elif i > split_number and j > split_number:
                both_above_pairs.append((p,[(i,j)]))
            else:
                split_pairs.append((p,[(i,j)]))
    
    both_below_pairs=update_fixed_coor(both_below_pairs,max_ligand_pairs=max_ligand_pairs)
    both_above_pairs=update_fixed_coor(both_above_pairs,max_ligand_pairs=max_ligand_pairs)
    split_pairs=update_fixed_coor(split_pairs,max_ligand_pairs=max_ligand_pairs)
    return both_below_pairs,both_above_pairs,split_pairs
    
def split_on_identifier(paired_test_triplets,split_number,max_ligand_pairs):
    train_data=[]
    val_data=[]
    for p,v in paired_test_triplets:
        for i in v:
            if i <= split_number:
                train_data.append((p,[i]))
            else:
                val_data.append((p,[i]))
    
    train_data=update_fixed_coor(train_data,max_ligand_pairs=max_ligand_pairs)
    val_data=update_fixed_coor(val_data,max_ligand_pairs=max_ligand_pairs)
    return train_data,val_data
                
def retrieve_data_from_folder(data_folder='Data/manuscript_data/',
                    training_folder='Data/training_data/',
                     max_ligands_pairs_per_prot=5,
                    absolute=True,
                    provided_targets=None,
                    temporal_ratio=-1.):
    
    if provided_targets is None:
        targets=next(os.walk(data_folder))[1]
    else:
        targets=provided_targets
        
    data_d=dict()
    for target in targets:
        trainingdata_folder=f'{training_folder}/{target}/'
        pdb_folder=f'{data_folder}{target}/pdbs'
        sdf_file=f'{data_folder}{target}/Selected_dockings.sdf'
        
        subd=next(os.walk(pdb_folder))[2]
        pdb_dict={'pdb_file':[], 'pdb_id': []}
        pdb_file_end='bloe.pdb'
        for d in subd:
                pdb_dict['pdb_file'].append(f'{pdb_folder}/{d}')
                if target=='BACE':
                    splitted_f=d.split('_',1)
                else:
                    splitted_f=d.rsplit('_',1)

                pdb_dict['pdb_id'].append(splitted_f[0])
                pdb_file_end=f'_{splitted_f[1]}'
        prot_df=pd.DataFrame(pdb_dict)

        protein_data=ResidueGranularityDataset( df=prot_df, df_file=None,
                 pdbs_col='pdb_id',
                 pdbs_file_col='pdb_file',
                  properties= None ,
                 root=f'{trainingdata_folder}ProteinDataFiles',
                 pdb2graph=protein_pdb2graph,
                 transform=None,
                 pre_transform=None, 
                 force_reload=True)

        soarse_mat_2dense=[('centriods_dists_edges_index','centriods_dists' )]

        pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        lig_data=atomistic3d_InMemoryDataset_SDFfile( sdf_file=sdf_file,
                         pdb_key='pdb',
                         properties= ['pIC50'] ,
                         root=f'{trainingdata_folder}LigandDataFiles',
                         sdf2graph=sdf2graph,
                         transform=None,
                         pre_transform=pre_transform, 
                         )
        sparse_mat_2dense=[('dist_mat_index','dist_mat_values' ),
                        ('topological_dist_mat_index','topological_dist_mat_values' ),
                                               ]

        interactiondata_file=f'{trainingdata_folder}absolute_interact_file.pt'
        if osp.exists(interactiondata_file):
            interaction_data=torch.load(interactiondata_file)
        else:
            interaction_data=InteractionDataset_SDFfile(
                             sdf_file=sdf_file,
                             pdb_key='pdb',
                             properties= ['pIC50'] ,
                             pdb_file_end=pdb_file_end,
                             n_cores=4,
                             chunksize=100)
            torch.save(interaction_data,interactiondata_file)
        prot_index_d={k:(v,[],[]) for v,k in zip(prot_df.index,prot_df.pdb_id)}
        
        jnj_id_to_index={}
        for idx,mol in enumerate(Chem.SDMolSupplier(sdf_file,removeHs=False)):
            pdb_nm=mol.GetProp('pdb')
            jnj_id=int(retrieve_prop_from_mol(mol, guesses=['JNJNUMBER','JNJ'], start='JNJ',remove_q=False))
            prot_index_d[pdb_nm][1].append(idx)
            prot_index_d[pdb_nm][2].append(jnj_id)
            jnj_id_to_index[jnj_id]=idx
        protein_coor=[ (v[0],v[1]) for k,v in prot_index_d.items()]
        
        split_jnj_number=-1
        if temporal_ratio>0.:
            all_jnj_ids=[]
            for k,v in prot_index_d.items():
                all_jnj_ids+=v[2]
            all_jnj_ids.sort()
            split_index=int(len(all_jnj_ids)*temporal_ratio)
            split_jnj_number=all_jnj_ids[split_index]
        
        both_above_data=None
        split_data=None
        if absolute:
            paired_test_triplets=generate_fixed_coor(protein_coor,max_ligand_pairs=max_ligands_pairs_per_prot)
            if split_jnj_number>0:
                train_indices,val_indices= split_on_identifier(paired_test_triplets,jnj_id_to_index[split_jnj_number],max_ligand_pairs=max_ligands_pairs_per_prot)
            else:
                train_indices,val_indices=train_test_split(paired_test_triplets, test_size=0.3, random_state=42)  
            train_data=CumulativeDataset(train_indices,protein_dataset=protein_data,ligand_dataset=lig_data, interaction_dataset=interaction_data)
            val_data=CumulativeDataset(val_indices,protein_dataset=protein_data,ligand_dataset=lig_data, interaction_dataset=interaction_data)    
        else:
            paired_test_triplets=generate_hierarchical_triplets(protein_coor,max_ligand_pairs=max_ligands_pairs_per_prot) 
            if len(paired_test_triplets) > 1:
                if split_jnj_number>0:
                    train_indices,both_above_pairs,split_pairs = split_pairs_on_identifier(paired_test_triplets,jnj_id_to_index[split_jnj_number],max_ligand_pairs=max_ligands_pairs_per_prot)
                    val_indices=both_above_pairs+split_pairs
                    both_above_data=CumulativePairedDataset(both_above_pairs,protein_dataset=protein_data,ligand_dataset=lig_data, interaction_dataset=interaction_data)
                    split_data=CumulativePairedDataset(split_pairs,protein_dataset=protein_data,ligand_dataset=lig_data, interaction_dataset=interaction_data) 
                else:
                    train_indices,val_indices=train_test_split(paired_test_triplets, test_size=0.15, random_state=42)  
            else:
                train_indices=[]
                val_indices=[]
            train_data=CumulativePairedDataset(train_indices,protein_dataset=protein_data,ligand_dataset=lig_data, interaction_dataset=interaction_data)
            val_data=CumulativePairedDataset(val_indices,protein_dataset=protein_data,ligand_dataset=lig_data, interaction_dataset=interaction_data) 
        
        print(f'{target} train data sz: {len(protein_coor)} complexes, {sum([len(idx[1]) for idx in train_indices])} ligand (s/pairs)')
        print(f'{target} val data sz: {len(protein_coor)} complexes, {sum([len(idx[1]) for idx in val_indices])} ligand (s/pairs)')
        
        if both_above_data is not None:
            data_d[target]={'train': train_data, 'validation':val_data, 'both_above_data':both_above_data, 'split_data':split_data}
        else:
            data_d[target]={'train': train_data, 'validation':val_data}
    return data_d


