import torch
import pandas as pd
import lightning as L
from torch.utils.data import random_split, DataLoader
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from collections import Counter

from  utils_3d.protein_features_util import protein_pdb2graph 
from  dataset_utils.dataset_3d import ResidueGranularityDataset
from  utils_3d.ligand_features_util import sdf2graph
from dataset_utils.dataset_3d import atomistic3d_InMemoryDataset
from  dataset_utils.dataset_3d import InteractionDataset
from  dataset_utils.dataset_3d import  CumulativeDataset,CumulativePairedDataset, alldata_collate_fn,fn_sparse2dense, fn_interaction_pad, collate_tensor_fn, generate_fixed_coor, update_fixed_coor
from dataset_utils.data_splitting import leak_proof_split, core_split, pdb_id_split ,fep2_ids, fep1_ids, cid_split, lk_cid_split

import os.path as osp
import os
from tqdm import tqdm
from rdkit import Chem

def get_pdbs_cids(data_folder='/domino/datasets/JnJInternal/manuscript_data/',provided_targets=None):
    
    if provided_targets is None:
        targets=next(os.walk(data_folder))[1]
    else:
        targets=provided_targets
        
    test_pdbs={}
    test_ids={}
    for target in targets:
        pdb_folder=f'{data_folder}{target}/pdbs'
        sdf_file=f'{data_folder}{target}/Selected_dockings.sdf'
        
        for mol in tqdm(Chem.SDMolSupplier(sdf_file,removeHs=False)):
            cid=mol.GetProp('ChEMBLID')
            pdb=mol.GetProp('pdb')
            #print(pdb,cid)
            test_ids[cid]=True
            test_pdbs[pdb]=True
        
    return [key for key,val in test_pdbs.items()],[key for key,val in test_ids.items()]

class AbsoluteDataModule(L.LightningDataModule):
    def __init__(self, config: dict[str,str] = None):
        super().__init__()
        self.prot_df= pd.read_csv(config['protein_df'], na_values = ['NAN', '?','NaN'])
        self.df= pd.read_csv(config['ligand_df'], na_values = ['NAN', '?','NaN'])
        if 'remove_pdbs' in config:
            for pdb_to_rm in config['remove_pdbs']:
                self.prot_df.loc[self.prot_df['pdb_id']==pdb_to_rm,'readable_pdb']=False
                self.df.loc[self.df['pdb_id']==pdb_to_rm,'readable_pdb']=False
                
        if 'readable_pdb' in self.prot_df:        
            self.prot_df=self.prot_df[self.prot_df['readable_pdb']].reset_index(drop=True)
        if 'readable_pdb' in self.df:   
            self.df=self.df[self.df['readable_pdb']].reset_index(drop=True)
        
        if 'pdb_id' not in self.df.columns:
            self.df['pdb_id']=[f.rsplit('/')[-2] for f in self.df['pdb_file']]
        if config['protein_ligand_mapping'] is None:
            self.protein_ind=[[i] for i in range(len(self.prot_df))]
        else:
            self.protein_ind=torch.load(config['protein_ligand_mapping'])
            
        self.processed_protein_folder=config['processed_protein_folder']
        self.processed_ligand_folder=config['processed_ligand_folder']
        self.interactiondata_file=config['interactiondata_file']
        self.n_cores=config['n_cores']
        self.config=config
        if not 'use_sampler' in self.config:
            self.config['use_sampler']=False
        assert 'pdb_id' in self.prot_df , 'pdb_id must be column in protein df'
        assert 'pdb_file' in self.prot_df , 'pdb_file must be column in protein df'
        assert 'pdb_id' in self.df , 'pdb_id must be column in ligand df'
        assert 'lig_file' in self.df , 'lig_file must be column in ligand df'
        assert 'ligand_affinity' in self.df , 'ligand_affinity must be column in ligand df'
        
    def prepare_data(self):
        # process proteins
        self.protein_data=ResidueGranularityDataset( df=self.prot_df,
                                     df_file=None,
                                     pdbs_col='pdb_id',
                                     pdbs_file_col='pdb_file',
                                     properties= None ,
                                     root=self.processed_protein_folder,
                                     pdb2graph=protein_pdb2graph,
                                     transform=None,
                                     pre_transform=None,
                                     n_cores=self.n_cores,
                                     force_reload=True)

        self.protein_collate_fn=[('centriods_dists_edges_index','centriods_dists' )]
        
        # process ligands
        pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.ligand_data=atomistic3d_InMemoryDataset( self.df,
                         pdbs_col='pdb_id',
                         ligand_file_col='lig_file',
                         properties= ['ligand_affinity'] ,
                         root=self.processed_ligand_folder,
                         sdf2graph=sdf2graph,
                         transform=None,
                         pre_transform=pre_transform, 
                         )
        self.ligand_collate_fn=[('dist_mat_index','dist_mat_values' ),
                        ('topological_dist_mat_index','topological_dist_mat_values' )
                                       ]
        #process interactions
        if osp.exists(self.interactiondata_file):
            self.interaction_data=torch.load(self.interactiondata_file)
        else:
            self.interaction_data=InteractionDataset(self.df,
                                     pdbs_col='pdb_id',
                                     pdbs_file_col='pdb_file',
                                     ligands_file_col='lig_file',
                                     properties= ['ligand_affinity'],
                                    n_cores=self.n_cores)
            torch.save(self.interaction_data,self.interactiondata_file)
        
        if 'empty_reactions' in self.config: 
            self.protein_coor=[]
            self.protein_ids=[]
            for i,v in enumerate(self.prot_df['pdb_id']):
                if v in self.config['empty_reactions']:
                    continue
                else:
                    self.protein_coor.append((i, [i]))
                    self.protein_ids.append(v)
        else:
            self.protein_ids=self.prot_df['pdb_id']    
            self.protein_coor=[ (p,l) for p,l in enumerate(self.protein_ind)]
        
        train_ind=[]
        val_ind=[]
        temp_ind=[]
        test_ind=[]
        if self.config['split_type']=='leakproof': #core #given pdbs
            train_ind,val_ind,test_ind = leak_proof_split(self.config['leakproof'],self.protein_coor,self.protein_ids,filters=self.config['filters'], add_unk_to_train=False)
            self.paired_triplets=generate_fixed_coor(train_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
            self.paired_val_triplets=generate_fixed_coor(val_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
        elif self.config['split_type']=='core': #core #given pdbs
            temp_ind,test_ind = core_split(self.config['leakproof'],self.protein_coor,self.protein_ids,filters=self.config['filters'], add_unk_to_train=False)
        elif self.config['split_type']=='stratified': #core #given pdbs
            temp_triplets=generate_fixed_coor(self.protein_coor,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
            labels=[ tri[0] for tri in temp_triplets]
            label_count=Counter(labels)
            new_labels=[ l if label_count[l]>2 else 'minority' for l in labels]
            train_val_trips,self.paired_test_triplets = train_test_split(temp_triplets, stratify=new_labels, test_size=self.config['test_ratio'], random_state=self.config['seed'])
            labels=[ tri[0] for tri in train_val_trips]
            label_count=Counter(labels)
            new_labels=[ l if label_count[l]>2 else 'minority' for l in labels]
            self.paired_triplets, self.paired_val_triplets=train_test_split(train_val_trips, stratify=new_labels, test_size=self.config['val_ratio'], random_state=self.config['seed']+self.config['seed'])
        elif self.config['split_type']=='lk_given_cids':
            if 'given_cids' not in self.config or self.config['given_cids'] is None:
                if 'manuscript_data_folder' in self.config:
                    _,given_cids=get_pdbs_cids(data_folder=self.config['manuscript_data_folder'])
                else:
                    _,given_cids=get_pdbs_cids(data_folder='/domino/datasets/JnJInternal/manuscript_data/')
            else:
                given_cids=config['given_cids']
            train_ind,val_ind,test_ind = lk_cid_split(self.config['leakproof'],self.df,given_cids,self.protein_coor,self.protein_ids,filters=self.config['filters'])
            self.paired_triplets=generate_fixed_coor(train_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
            self.paired_val_triplets=generate_fixed_coor(val_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
        elif self.config['split_type']=='given_cids':
            if 'given_cids' not in self.config or self.config['given_cids'] is None:
                if 'manuscript_data_folder' in self.config:
                    _,given_cids=get_pdbs_cids(data_folder=self.config['manuscript_data_folder'])
                else:
                    _,given_cids=get_pdbs_cids(data_folder='/domino/datasets/JnJInternal/manuscript_data/')
            else:
                given_cids=config['given_cids']
            temp_ind,test_ind = cid_split(self.df,given_cids,self.protein_coor,self.protein_ids)
        else:
            if len(self.config['test_pdbs'])>0:
                test_pdbs=self.config['test_pdbs']
            else:
                test_pdbs=fep2_ids+fep1_ids
            temp_ind,test_ind = pdb_id_split(self.protein_coor,self.protein_ids,pdb_ids=test_pdbs)
            
        if len(test_ind)>0:
            self.paired_test_triplets=generate_fixed_coor(test_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
        
        if len(temp_ind)>0:
            if self.config['val_split_type']=='leave_protein_out':
                generator = torch.Generator().manual_seed(self.config['seed'])
                train_ind,val_ind=random_split(temp_ind, [1.0-self.config['val_ratio'],self.config['val_ratio']], generator=generator)
                self.paired_triplets=generate_fixed_coor(train_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
                self.paired_val_triplets=generate_fixed_coor(val_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
            else:
                temp_triplets=generate_fixed_coor(temp_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
                train_ind,val_ind= [],[]
                generator = torch.Generator().manual_seed(self.config['seed'])
                splits=random_split(temp_triplets, [1.0-self.config['val_ratio'],self.config['val_ratio']], generator=generator)
                self.paired_triplets=list(splits[0])
                self.paired_val_triplets=list(splits[1])
                
        print(f'Training data sz: {len(train_ind)} complexes, {sum([len(idx[1]) for idx in self.paired_triplets])} pairs')
        print(f'Validation data sz: {len(val_ind)} complexes, {sum([len(idx[1]) for idx in self.paired_val_triplets])} pairs')
        print(f'Test data sz:  {len(test_ind)} complexes, {sum([len(idx[1]) for idx in self.paired_test_triplets])} pairs')
        
        self.list_keys=['ligands','interactions','indices']

        #for each key in the data, provide dict with collate function ('fn') and possible params ('params')
        self.alldata_params={'protein': {'fn': fn_sparse2dense,'params':self.protein_collate_fn},
                        'protein_indices': {'fn':collate_tensor_fn},
                        'ligands': {'fn':fn_sparse2dense,'params':self.ligand_collate_fn},
                        'interactions': {'fn':fn_interaction_pad},
                        'indices': {'fn':collate_tensor_fn}}
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.traindata=CumulativeDataset(self.paired_triplets,protein_dataset=self.protein_data,ligand_dataset=self.ligand_data, interaction_dataset=self.interaction_data)
            self.valdata=CumulativeDataset(self.paired_val_triplets,protein_dataset=self.protein_data,ligand_dataset=self.ligand_data, interaction_dataset=self.interaction_data)   
            self.traindata.test_i_indices(10)
            self.valdata.test_i_indices(10)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.testdata =CumulativeDataset(self.paired_test_triplets,protein_dataset=self.protein_data,ligand_dataset=self.ligand_data, interaction_dataset=self.interaction_data)
            self.testdata.test_i_indices(10)  

        if stage == "predict":
            self.testdata =CumulativeDataset(self.paired_test_triplets,protein_dataset=self.protein_data,ligand_dataset=self.ligand_data, interaction_dataset=self.interaction_data) 
            self.testdata.test_i_indices(10)
        
    def train_dataloader(self):
        self.traindata.index_coor=update_fixed_coor(self.paired_triplets,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
        if self.config['use_sampler']:
            weights = self.traindata.generate_weights(prop='ligand_affinity', eps=1e-3)                                                
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 
            return DataLoader(self.traindata, batch_size=self.config['batch_size'], sampler = sampler, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )
        else:
            return DataLoader(self.traindata, batch_size=self.config['batch_size'], shuffle=True, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )
            

    def val_dataloader(self):
        return DataLoader(self.valdata, batch_size=self.config['batch_size'], shuffle=False, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )

    def test_dataloader(self):
        return DataLoader(self.testdata, batch_size=self.config['batch_size'], shuffle=False, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )

    def predict_dataloader(self):
        return DataLoader(self.testdata, batch_size=self.config['batch_size'], shuffle=False, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )


class RelativeDataModule(L.LightningDataModule):
    def __init__(self, config: dict[str,str] = None):
        super().__init__()
        self.prot_df= pd.read_csv(config['protein_df'], na_values = ['NAN', '?','NaN'])
        self.df= pd.read_csv(config['ligand_df'], na_values = ['NAN', '?','NaN'])
        self.df['pdb_id']=[f.rsplit('/')[-2] for f in self.df['pdb_file']]
        self.protein_ind=torch.load(config['protein_ligand_mapping'])
        self.processed_protein_folder=config['processed_protein_folder']
        self.processed_ligand_folder=config['processed_ligand_folder']
        self.interactiondata_file=config['interactiondata_file']
        self.n_cores=config['n_cores']
        self.config=config
        if not 'use_sampler' in self.config:
            self.config['use_sampler']=False
        assert 'pdb_id' in self.prot_df , 'pdb_id must be column in protein df'
        assert 'pdb_file' in self.prot_df , 'pdb_file must be column in protein df'
        assert 'pdb_id' in self.df , 'pdb_id must be column in ligand df'
        assert 'lig_file' in self.df , 'lig_file must be column in ligand df'
        assert 'ligand_affinity' in self.df , 'ligand_affinity must be column in ligand df'
        
    def prepare_data(self):
        # process proteins
        self.protein_data=ResidueGranularityDataset( df=self.prot_df,
                                     df_file=None,
                                     pdbs_col='pdb_id',
                                     pdbs_file_col='pdb_file',
                                     properties= None ,
                                     root=self.processed_protein_folder,
                                     pdb2graph=protein_pdb2graph,
                                     transform=None,
                                     pre_transform=None,
                                     n_cores=self.n_cores,
                                     force_reload=True)

        self.protein_collate_fn=[('centriods_dists_edges_index','centriods_dists' )]
        
        # process ligands
        pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.ligand_data=atomistic3d_InMemoryDataset( self.df,
                         pdbs_col='pdb_id',
                         ligand_file_col='lig_file',
                         properties= ['ligand_affinity'] ,
                         root=self.processed_ligand_folder,
                         sdf2graph=sdf2graph,
                         transform=None,
                         pre_transform=pre_transform, 
                         )
        self.ligand_collate_fn=[('dist_mat_index','dist_mat_values' ),
                        ('topological_dist_mat_index','topological_dist_mat_values' )
                                       ]
        #process interactions
        if osp.exists(self.interactiondata_file):
            self.interaction_data=torch.load(self.interactiondata_file)
        else:
            self.interaction_data=InteractionDataset(self.df,
                                     pdbs_col='pdb_id',
                                     pdbs_file_col='pdb_file',
                                     ligands_file_col='lig_file',
                                     properties= ['ligand_affinity'],
                                    n_cores=self.n_cores)
            torch.save(self.interaction_data,self.interactiondata_file)
        
        self.protein_coor=[ (p,l) for p,l in enumerate(self.protein_ind)]
        temp_ind=[]
        test_ind=[]
        if self.config['split_type']=='leakproof': #core #given pdbs
            train_ind,val_ind,test_ind = leak_proof_split(self.config['leakproof'],self.protein_coor,self.prot_df['pdb_id'],filters=self.config['filters'], add_unk_to_train=False)
            self.paired_triplets=generate_fixed_coor(train_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
            self.paired_val_triplets=generate_fixed_coor(val_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
        elif self.config['split_type']=='core': #core #given pdbs
            temp_ind,test_ind = core_split(self.config['leakproof'],self.protein_coor,self.prot_df['pdb_id'],filters=self.config['filters'], add_unk_to_train=False)
        elif self.config['split_type']=='stratified': #core #given pdbs
            train_ind,val_ind= [],[]
            temp_triplets=generate_fixed_coor(self.protein_coor,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
            labels=[ tri[0] for tri in temp_triplets]
            label_count=Counter(labels)
            new_labels=[ l if label_count[l]>2 else 'minority' for l in labels]
            train_val_trips,self.paired_test_triplets = train_test_split(temp_triplets, stratify=new_labels, test_size=self.config['test_ratio'], random_state=self.config['seed'])
            labels=[ tri[0] for tri in train_val_trips]
            label_count=Counter(labels)
            new_labels=[ l if label_count[l]>2 else 'minority' for l in labels]
            self.paired_triplets, self.paired_val_triplets=train_test_split(train_val_trips, stratify=new_labels, test_size=self.config['val_ratio'], random_state=self.config['seed']+self.config['seed'])
        elif self.config['split_type']=='lk_given_cids':
            if 'given_cids' not in self.config or self.config['given_cids'] is None:
                if 'manuscript_data_folder' in self.config:
                    _,given_cids=get_pdbs_cids(data_folder=self.config['manuscript_data_folder'])
                else:
                    _,given_cids=get_pdbs_cids(data_folder='/domino/datasets/JnJInternal/manuscript_data/')
            else:
                given_cids=config['given_cids']
            train_ind,val_ind,test_ind = lk_cid_split(self.config['leakproof'],self.df,given_cids,self.protein_coor,self.prot_df['pdb_id'],filters=self.config['filters'])
            self.paired_triplets=generate_fixed_coor(train_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
            self.paired_val_triplets=generate_fixed_coor(val_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
        elif self.config['split_type']=='given_cids':
            if 'given_cids' not in self.config or self.config['given_cids'] is None:
                if 'manuscript_data_folder' in self.config:
                    _,given_cids=get_pdbs_cids(data_folder=self.config['manuscript_data_folder'])
                else:
                    _,given_cids=get_pdbs_cids(data_folder='/domino/datasets/JnJInternal/manuscript_data/')
            else:
                given_cids=self.config['given_cids']
            temp_ind,test_ind = cid_split(self.df,given_cids,self.protein_coor,self.prot_df['pdb_id'])
            
        else:
            if len(self.config['test_pdbs'])>0:
                test_pdbs=self.config['test_pdbs']
            else:
                test_pdbs=fep2_ids+fep1_ids
            temp_ind,test_ind = pdb_id_split(self.protein_coor,self.prot_df['pdb_id'],pdb_ids=test_pdbs)
            
        if len(test_ind)>0:
            self.paired_test_triplets=generate_fixed_coor(test_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
        
        if len(temp_ind)>0:
            if self.config['val_split_type']=='leave_protein_out':
                generator = torch.Generator().manual_seed(self.config['seed'])
                train_ind,val_ind=random_split(temp_ind, [1.0-self.config['val_ratio'],self.config['val_ratio']], generator=generator)
                self.paired_triplets=generate_fixed_coor(train_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
                self.paired_val_triplets=generate_fixed_coor(val_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
            else:
                temp_triplets=generate_fixed_coor(temp_ind,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
                train_ind,val_ind= [],[]
                generator = torch.Generator().manual_seed(self.config['seed'])
                splits=random_split(temp_triplets, [1.0-self.config['val_ratio'],self.config['val_ratio']], generator=generator)
                self.paired_triplets=list(splits[0])
                self.paired_val_triplets=list(splits[1])
                
        print(f'Training data sz: {len(train_ind)} complexes, {sum([len(idx[1]) for idx in self.paired_triplets])} pairs')
        print(f'Validation data sz: {len(val_ind)} complexes, {sum([len(idx[1]) for idx in self.paired_val_triplets])} pairs')
        print(f'Test data sz:  {len(test_ind)} complexes, {sum([len(idx[1]) for idx in self.paired_test_triplets])} pairs')
        
        self.list_keys=['ligands_one','interactions_one','indices_one','ligands_two','interactions_two','indices_two']

        #for each key in the data, provide dict with collate function ('fn') and possible params ('params')
        self.alldata_params={'protein': {'fn': fn_sparse2dense,'params': self.protein_collate_fn},
                        'protein_indices': {'fn':collate_tensor_fn},
                        'ligands_one': {'fn':fn_sparse2dense,'params':self.ligand_collate_fn},
                        'interactions_one': {'fn':fn_interaction_pad},
                        'indices_one': {'fn':collate_tensor_fn},
                        'ligands_two': {'fn':fn_sparse2dense,'params':self.ligand_collate_fn},
                        'interactions_two': {'fn':fn_interaction_pad},
                        'indices_two': {'fn':collate_tensor_fn}}
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.traindata=CumulativePairedDataset(self.paired_triplets,protein_dataset=self.protein_data,ligand_dataset=self.ligand_data, interaction_dataset=self.interaction_data)
            self.valdata=CumulativePairedDataset(self.paired_val_triplets,protein_dataset=self.protein_data,ligand_dataset=self.ligand_data, interaction_dataset=self.interaction_data)   
            self.traindata.test_i_indices(10)
            self.valdata.test_i_indices(10)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.testdata =CumulativePairedDataset(self.paired_test_triplets,protein_dataset=self.protein_data,ligand_dataset=self.ligand_data, interaction_dataset=self.interaction_data)
            self.testdata.test_i_indices(10)  

        if stage == "predict":
            self.testdata =CumulativePairedDataset(self.paired_test_triplets,protein_dataset=self.protein_data,ligand_dataset=self.ligand_data, interaction_dataset=self.interaction_data) 
            self.testdata.test_i_indices(10)
            

    def train_dataloader(self):
        self.traindata.index_triplets=update_fixed_coor(self.paired_triplets,max_ligand_pairs=self.config['max_ligands_pairs_per_prot'])
        if self.config['use_sampler']:
            weights = self.traindata.generate_weights(prop='ligand_affinity', eps=1e-3)                                                
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 
            return DataLoader(self.traindata, batch_size=self.config['batch_size'], sampler = sampler, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )
        else:
            return DataLoader(self.traindata, batch_size=self.config['batch_size'], shuffle=True, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )
            

    def val_dataloader(self):
        return DataLoader(self.valdata, batch_size=self.config['batch_size'], shuffle=False, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )

    def test_dataloader(self):
        return DataLoader(self.testdata, batch_size=self.config['batch_size'], shuffle=False, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )

    def predict_dataloader(self):
        return DataLoader(self.testdata, batch_size=self.config['batch_size'], shuffle=False, collate_fn=lambda data_list: alldata_collate_fn(data_list,self.list_keys, self.alldata_params) )