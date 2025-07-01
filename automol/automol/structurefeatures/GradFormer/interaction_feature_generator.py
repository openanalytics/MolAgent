import os
from  automol.structurefeatures.GradFormer.utils_3d.protein_features_util import protein_pdb2graph 
from  automol.structurefeatures.GradFormer.dataset_utils.dataset_3d import ResidueGranularityDataset
import pandas as pd
import torch_geometric.transforms as T
from  automol.structurefeatures.GradFormer.utils_3d.ligand_features_util import sdf2graph
from automol.structurefeatures.GradFormer.dataset_utils.dataset_3d import atomistic3d_InMemoryDataset_SDFfile
from  automol.structurefeatures.GradFormer.dataset_utils.dataset_3d import InteractionDataset_SDFfile
import os.path as osp
import torch
from rdkit import Chem
from automol.structurefeatures.GradFormer.dataset_utils.dataset_3d import CumulativeDataset, alldata_collate_fn,fn_sparse2dense, fn_interaction_pad, collate_tensor_fn, generate_fixed_coor
from automol.structurefeatures.GradFormer.dataset_utils.dataset_3d import CumulativePairedDataset, generate_hierarchical_triplets,update_fixed_coor, retrieve_prop_from_mol, update_fixed_coor
from automol.structurefeatures.GradFormer.dataset_utils.dataset_3d import  alldata_collate_fn, fn_sparse2dense, collate_tensor_fn, fn_interaction_pad
from torch.utils.data import DataLoader
from automol.feature_generators import FeatureGenerator

from automol.structurefeatures.GradFormer.building_elements_encoders import  mol_atomstic_encoder, mol_residues_level_encoder, mol_residues_light_level_encoder,mol_light_atomstic_encoder,mol_residue_atomstic_encoder
from automol.structurefeatures.GradFormer.interaction_encoders import InteractionAbsoluteGraphEncoder,FullParameterDistanceEmbedding, FullParameterEmbedding,InteractionAbsolutePharmacoPhoreEncoder,InteractionAbsoluteLigandEncoder
import torch_geometric.transforms as T
from automol.structurefeatures.GradFormer.lightning_trainer import MHAAbsolutehead, MHAAbsolutePharmaRegressionhead, LinearAbsolutehead, MHARelativePharmaRegressionhead,MHARelativeDecoderhead,LinearRelativehead

import sys
sys.modules['interaction_encoders'] = sys.modules['automol.structurefeatures.GradFormer.interaction_encoders']
sys.modules['building_elements_encoders'] = sys.modules['automol.structurefeatures.GradFormer.building_elements_encoders']
sys.modules['lightning_trainer'] = sys.modules['automol.structurefeatures.GradFormer.lightning_trainer']
sys.modules['grad_conv'] = sys.modules['automol.structurefeatures.GradFormer.grad_conv']
sys.modules['utils_3d'] = sys.modules['automol.structurefeatures.GradFormer.utils_3d']
sys.modules['dataset_utils'] = sys.modules['automol.structurefeatures.GradFormer.dataset_utils']

class InteractionFeaturesGenerator(FeatureGenerator):
    """
    feature generator returning the interaction features
    """

    def __init__(self,pdb_folder='',
                    sdf_file='',
                    processed_data_folder='',
                     config=None,
                     name='ABCD',
                     model_f=f'model.ckpt',
                     properties= ['pIC50'] ,
                     nb_features=132):
        """
        Initialization 
        """
        trainingdata_folder=f'{processed_data_folder}/{name}/'
        self.properties=properties
        self._setup_data(pdb_folder,sdf_file,trainingdata_folder,name)
        self._setup_model(config,model_f)
        ## list of names of the features
        self.names= [f'automol_interaction_feature_{name}_{i}' for i in range(nb_features)]
        ## number of features
        self.nb_features=nb_features
        self.properties=properties
        ## generator name
        self.generator_name=f'automol_interaction_feature_{name}'
        
    def _setup_model(self,config,model_f):

        if config['pure_pharma']:
            head=MHAAbsolutePharmaRegressionhead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)# 

        elif config['ligand_res_enc']:
            if config['simple_head']:
                head=LinearAbsolutehead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#
            else:
                head=MHAAbsolutehead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#
        else:
            if config['simple_head']:
                head=LinearAbsolutehead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#
            else:
                head=MHAAbsolutehead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#
                
        self.model=head

        
    def _setup_data(self,pdb_folder,sdf_file,trainingdata_folder,name):
                
        subd=next(os.walk(pdb_folder))[2]
        pdb_dict={'pdb_file':[], 'pdb_id': []}
        pdb_file_end='bloe.pdb'
        for pdb_idx,d in enumerate(subd):
                pdb_dict['pdb_file'].append(f'{pdb_folder}/{d}')
                if name=='BACE':
                    splitted_f=d.split('_',1)
                else:
                    splitted_f=d.rsplit('_',1)

                pdb_dict['pdb_id'].append(splitted_f[0])
                pdb_file_end=f'_{splitted_f[1]}'
        prot_df=pd.DataFrame(pdb_dict)

        self.protein_data=ResidueGranularityDataset( df=prot_df, df_file=None,
                 pdbs_col='pdb_id',
                 pdbs_file_col='pdb_file',
                  properties= None ,
                 root=f'{trainingdata_folder}ProteinDataFiles',
                 pdb2graph=protein_pdb2graph,
                 transform=None,
                 pre_transform=None, 
                 force_reload=True)

        self.protein_collate_fn=[('centriods_dists_edges_index','centriods_dists' )]

        pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.lig_data=atomistic3d_InMemoryDataset_SDFfile( sdf_file=sdf_file,
                         pdb_key='pdb',
                         properties= self.properties ,
                         root=f'{trainingdata_folder}LigandDataFiles',
                         sdf2graph=sdf2graph,
                         transform=None,
                         pre_transform=pre_transform, 
                         )
        self.ligand_collate_fn=[('dist_mat_index','dist_mat_values' ),
                        ('topological_dist_mat_index','topological_dist_mat_values' ),
                                               ]

        interactiondata_file=f'{trainingdata_folder}absolute_interact_file.pt'
        if osp.exists(interactiondata_file):
            self.interaction_data=torch.load(interactiondata_file,weights_only=False)
        else:
            self.interaction_data=InteractionDataset_SDFfile(
                             pdb_file_folder=pdb_folder,
                             sdf_file=sdf_file,
                             pdb_key='pdb',
                             properties= self.properties ,
                             pdb_file_end=pdb_file_end,
                             n_cores=4,
                             chunksize=100)
            torch.save(self.interaction_data,interactiondata_file)
        self.prot_index_d={k:int(v) for v,k in zip(prot_df.index,prot_df.pdb_id)}
        
        self.ligandindex_to_pdbindex=[]
        self.ligandsmiles_to_ligandindex={}
        for idx,mol in enumerate(Chem.SDMolSupplier(sdf_file,removeHs=False)):
            from automol.standardize import standardize
            pdb_nm=mol.GetProp('pdb')
            self.ligandindex_to_pdbindex.append(self.prot_index_d[pdb_nm])
            self.ligandsmiles_to_ligandindex[standardize(Chem.MolToSmiles(mol))]=idx
        
    def generate(self,smiles):
        ligand_idx=[self.ligandsmiles_to_ligandindex[smi] for smi in smiles]
        
        list_keys=['ligands','interactions','indices']

        #for each key in the data, provide dict with collate function ('fn') and possible params ('params')
        alldata_params={'protein': {'fn': fn_sparse2dense,'params':self.protein_collate_fn},
                    'protein_indices': {'fn':collate_tensor_fn},
                    'ligands': {'fn':fn_sparse2dense,'params':self.ligand_collate_fn},
                    'interactions': {'fn':fn_interaction_pad},
                    'indices': {'fn':collate_tensor_fn}}

        paired_triplets=[ (self.ligandindex_to_pdbindex[v],[v]) for v in ligand_idx]  
        data_set=CumulativeDataset(paired_triplets,protein_dataset=self.protein_data,ligand_dataset=self.lig_data, interaction_dataset=self.interaction_data)
            
        data_loader = DataLoader(data_set, batch_size=3, shuffle=False, collate_fn=lambda data_list: alldata_collate_fn(data_list,list_keys, alldata_params) )
        X_l=[]
        with torch.no_grad():
            self.model.eval()
            for idx,batch_data in enumerate(data_loader):
                #batch_data=to_cuda(batch_data)
                out=self.model.generate_features(batch_data)
                X_l+=out.float().cpu().detach()
                
        return torch.vstack(X_l).numpy()


#################
#Ligand index to pdb index
#init protein,ligand and interaction dataset for all
# provide indices, map to protein, create cumulative dataset and run model
################
class InteractionRelativeFeaturesGenerator(FeatureGenerator):       
    def __init__(self,pdb_folder='',
                    sdf_file='',
                    processed_data_folder='',
                     config=None,
                     name='ABCD',
                     model_f=f'model.ckpt',
                     properties= ['pIC50'] ,
                     nb_features=132):
        """
        Initialization 
        """
        trainingdata_folder=f'{processed_data_folder}/{name}/'
        self.properties=properties
        self._setup_data(pdb_folder,sdf_file,trainingdata_folder,name)
        self._setup_model(config,model_f)
        ## list of names of the features
        self.names= [f'automol_interaction_feature_{name}_{i}' for i in range(nb_features)]
        ## number of features
        self.nb_features=nb_features
        ## generator name
        self.generator_name=f'automol_interaction_feature_{name}'
        
    def _setup_model(self,config,model_f):
        
        if config['pure_pharma']:
            head=MHARelativePharmaRegressionhead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#

        elif config['ligand_res_enc']:
            if config['simple_head']:
                head=LinearRelativehead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#
            else:
                head=MHARelativeDecoderhead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#
        else:
            if config['simple_head']:
                head=LinearRelativehead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#
            else:
                head=MHARelativeDecoderhead.load_from_checkpoint(model_f)#torch.load(model_f, map_location=torch.device("cpu"), weights_only=True)#
                
        self.model=head

        
    def _setup_data(self,pdb_folder,sdf_file,trainingdata_folder,name):
        
        
        subd=next(os.walk(pdb_folder))[2]
        pdb_dict={'pdb_file':[], 'pdb_id': []}
        pdb_file_end='bloe.pdb'
        for pdb_idx,d in enumerate(subd):
                pdb_dict['pdb_file'].append(f'{pdb_folder}/{d}')
                if name=='BACE':
                    splitted_f=d.split('_',1)
                else:
                    splitted_f=d.rsplit('_',1)

                pdb_dict['pdb_id'].append(splitted_f[0])
                pdb_file_end=f'_{splitted_f[1]}'
        prot_df=pd.DataFrame(pdb_dict)

        self.protein_data=ResidueGranularityDataset( df=prot_df, df_file=None,
                 pdbs_col='pdb_id',
                 pdbs_file_col='pdb_file',
                  properties= None ,
                 root=f'{trainingdata_folder}ProteinDataFiles',
                 pdb2graph=protein_pdb2graph,
                 transform=None,
                 pre_transform=None, 
                 force_reload=True)

        self.protein_collate_fn=[('centriods_dists_edges_index','centriods_dists' )]

        pre_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.lig_data=atomistic3d_InMemoryDataset_SDFfile( sdf_file=sdf_file,
                         pdb_key='pdb',
                         properties= self.properties ,
                         root=f'{trainingdata_folder}LigandDataFiles',
                         sdf2graph=sdf2graph,
                         transform=None,
                         pre_transform=pre_transform, 
                         )
        self.ligand_collate_fn=[('dist_mat_index','dist_mat_values' ),
                        ('topological_dist_mat_index','topological_dist_mat_values' ),
                                               ]

        interactiondata_file=f'{trainingdata_folder}absolute_interact_file.pt'
        if osp.exists(interactiondata_file):
            self.interaction_data=torch.load(interactiondata_file,weights_only=False)
        else:
            self.interaction_data=InteractionDataset_SDFfile(
                             sdf_file=sdf_file,
                             pdb_file_folder=pdb_folder,
                             pdb_key='pdb',
                             properties= self.properties ,
                             pdb_file_end=pdb_file_end,
                             n_cores=4,
                             chunksize=100)
            torch.save(self.interaction_data,interactiondata_file)
        self.prot_index_d={k:int(v) for v,k in zip(prot_df.index,prot_df.pdb_id)}
        
        self.ligandindex_to_pdbindex=[]
        self.ligandsmiles_to_ligandindex={}
        for idx,mol in enumerate(Chem.SDMolSupplier(sdf_file,removeHs=False)):
            from automol.standardize import standardize
            pdb_nm=mol.GetProp('pdb')
            self.ligandindex_to_pdbindex.append(self.prot_index_d[pdb_nm])
            self.ligandsmiles_to_ligandindex[standardize(Chem.MolToSmiles(mol))]=idx
            
            
    def generate_w_pairs(self,smiles,original_indices,new_indices):
        list_keys=['ligands_one','interactions_one','indices_one','ligands_two','interactions_two','indices_two']

        alldata_params={'protein': {'fn': fn_sparse2dense,'params': self.protein_collate_fn},
                        'protein_indices': {'fn':collate_tensor_fn},
                        'ligands_one': {'fn':fn_sparse2dense,'params':self.ligand_collate_fn},
                        'interactions_one': {'fn':fn_interaction_pad},
                        'indices_one': {'fn':collate_tensor_fn},
                        'ligands_two': {'fn':fn_sparse2dense,'params':self.ligand_collate_fn},
                        'interactions_two': {'fn':fn_interaction_pad},
                        'indices_two': {'fn':collate_tensor_fn}
                       }

        paired_triplets=[ (self.ligandindex_to_pdbindex[v[0]],[v]) for v in original_indices]  
        data_set=CumulativePairedDataset(paired_triplets,protein_dataset=self.protein_data,ligand_dataset=self.lig_data, interaction_dataset=self.interaction_data)
            
        data_loader = DataLoader(data_set, batch_size=3, shuffle=False, collate_fn=lambda data_list: alldata_collate_fn(data_list,list_keys, alldata_params) )
        X_l=[]
        with torch.no_grad():
            self.model.eval()
            for idx,batch_data in enumerate(data_loader):
                #batch_data=to_cuda(batch_data)
                out=self.model.generate_features(batch_data)
                X_l+=out.float().cpu().detach()
                
        return torch.vstack(X_l).numpy()
    
    
    
    

    