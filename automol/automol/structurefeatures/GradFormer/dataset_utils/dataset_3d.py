import torch
from torch.utils.data import Dataset
import itertools
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
#from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
#from utils import process_sph
from torch.utils.data import  Dataset
import pyximport
#add_path=os.path.realpath(__file__)
#sys.path.append(add_path)
pyximport.install(setup_args={'include_dirs': np.get_include()})
#from add_path import algos
from torch_geometric.data import InMemoryDataset
import os
import os.path as osp
from rdkit import Chem

#####################
from automol.structurefeatures.GradFormer.utils_3d.ligand_features_util import sdf2graph, mol2graph , smiles2graph
from automol.structurefeatures.GradFormer.utils_3d.residue import Molecule
from automol.structurefeatures.GradFormer.utils_3d.get_interactions import interactions_2graph
############################################
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

from typing import List

from torch_geometric.utils import dense_to_sparse, to_dense_adj

from tqdm import tqdm

###############
def  mol2_pyg(mol):
                data = Data()
                graph=mol2graph(mol)
                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])
    
                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to( torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                #data.y = torch.Tensor([y])
                #y = data_df.iloc[i][properties]
                data['sasa_atom_wise']= torch.from_numpy(graph['sasa_atom_wise']).float()
                data['dist_mat_index'], data['dist_mat_values'] =dense_to_sparse(torch.from_numpy(graph['dist_mat']).float())
                data['topological_dist_mat_index'], data['topological_dist_mat_values'] =dense_to_sparse(torch.from_numpy(graph['topological_dist_mat']).float())
                
                # to recover dense:
                #ds=to_dense_adj(edge_index=data['dist_mat_index'] , edge_attr=data['dist_mat_values'] ).squeeze(0)
                #torch.all(torch.eq(ds,dist_mat))
                return data
##########################################################################
class atomistic3d_InMemoryDataset(InMemoryDataset):
    def __init__(self, df=None, df_file=None,
                 pdbs_col='pdb_id',
                 ligand_file_col='lig_file',
                 properties= ['A','B'] ,
                 root='temp_datasets',
                 sdf2graph=sdf2graph,
                 transform=None,
                 pre_transform=None, 
                 force_reload=True):
        self.df_file = df_file
        self.df = df
        if self.df_file: 
            self.df = pd.read_csv(self.df_file)
            df.reset_index(drop=True, inplace=True)
        assert isinstance(self.df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        assert pdbs_col in self.df.columns , f'{pdbs_col} is not in the columns of {self.df}'
        assert ligand_file_col in self.df.columns , f'{ligand_file_col} is not in the columns of {self.df}'
        self.pdbs_col= pdbs_col
        self.ligand_file_col=ligand_file_col
        self.properties=properties
        if self.properties:
            for pro in self.properties:  assert pro in self.df.columns , f'{pro} is not in the columns of df'
    
        self.root= osp.join(root)
        self.sdf2graph = sdf2graph
        super().__init__(root=self.root,  transform = transform, pre_transform=pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return df_file

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        pdbs_list = self.df[self.pdbs_col]
        ligand_files=self.df[self.ligand_file_col]
        assert(len(pdbs_list)==len(ligand_files))

        print('Converting mol strings into graphs...')
        data_list = []
        for i in tqdm(range(len(pdbs_list))):
            
                #try:            
                pdb = pdbs_list[i]
                sdf_file=ligand_files[i]

                #graph = self.sdf2graph(sdf_file)
                mol= Chem.SDMolSupplier(sdf_file,removeHs=False)[0]
                data=mol2_pyg(mol)
                data.pdb=pdb
                for pro in self.properties: data[pro]=torch.Tensor([self.df.iloc[i][pro]])#.to(torch.float)
                if self.pre_transform: data=self.pre_transform(data)
                data_list.append(data)
            
        print('Saving...')
        self.save(data_list, self.processed_paths[0])
        
        
def retrieve_prop_from_mol(mol,*, guesses:List[str], start:str,remove_q:bool=True):
    val=None
    prop_dict=mol.GetPropsAsDict()
    for p in guesses:
        if p in prop_dict:
            val=mol.GetProp(p)
            break
    if val is None:
        for key in prop_dict.keys():
            if key.startswith(start):
                val=mol.GetProp(key)
                break
    if remove_q:
        if val[0]=='<' or val[0]=='>':
            val=val[1:]
        if val[0]=='=':
            val=val[1:]
    return val

        
class atomistic3d_InMemoryDataset_SDFfile(InMemoryDataset):
    def __init__(self, sdf_file='ligands.sdf',
                 pdb_key='pdb',
                 properties= ['A','B'] ,
                 root='temp_datasets',
                 sdf2graph=sdf2graph,
                 transform=None,
                 pre_transform=None, 
                 force_reload=True):
        
        self.sdf_file=sdf_file
        self.pdb_key=pdb_key
        self.properties=properties
        self.root= osp.join(root)
        self.sdf2graph = sdf2graph
        super().__init__(root=self.root,  transform = transform, pre_transform=pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return df_file

    @property
    def processed_file_names(self):
        return ['ligand_data_processed.pt']

    def process(self):

        print('Converting mol strings into graphs...')
        data_list = []
        for mol in tqdm(Chem.SDMolSupplier(self.sdf_file,removeHs=False)):
            pdb_nm=mol.GetProp(self.pdb_key)
            data=mol2_pyg(mol)
            data.pdb=pdb_nm
            for pro in self.properties: 
                if pro=='pIC50':
                    data['ligand_affinity']=torch.Tensor([float(retrieve_prop_from_mol(mol,guesses=[pro],start=pro)) ])                
                else:
                    data[pro]=torch.Tensor([float(retrieve_prop_from_mol(mol,guesses=[pro],start=pro)) ])#.to(torch.float)            
            if self.pre_transform: data=self.pre_transform(data)
            data_list.append(data)
            
        print('Saving...')
        self.save(data_list, self.processed_paths[0])

##########################################
from automol.structurefeatures.GradFormer.utils_3d.protein_features_util import protein_mol2graph , protein_pdb2graph
###############
def protein_mol2pyg(protein_mol):
                data = Data()
                graph= protein_mol2graph(protein_mol)
                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.__num_nodes__ = int(graph['num_nodes'])
                data.covalent_edge_index = torch.from_numpy(graph['edge_index']).to( torch.int64)
                data.covalent_edge_attr = torch.from_numpy(graph['edge_feat']).to( torch.int64)
                data.noncovalent_edge_index = torch.from_numpy(graph['noncovalent_edge_index']).to( torch.int64)
                data.noncovalent_edge_attr = torch.from_numpy(graph['noncovalent_edge_feat']).to( torch.int64)
                ## get total edge indices here to enable merging later 
                data.edge_index=torch.cat((data.covalent_edge_index,data.noncovalent_edge_index),-1)
                data['res_res_BB_dihs']= torch.from_numpy(graph['res_res_BB_dihs']).to(torch.float)
                data['rotamers']= torch.from_numpy(graph['rotamers']).to(torch.float)
                data['sasa_residue_wise']= torch.from_numpy(graph['sasa_residue_wise']).to(torch.float)
                data['centriods_dists_edges_index'] = torch.from_numpy(graph['centriods_dists_edges_index']).to( torch.int64)
                data['centriods_dists']= torch.from_numpy(graph['centriods_dists']).to(torch.float)
                data['residue_ids']=torch.from_numpy(graph['residue_ids']).to( torch.int64)
                data['chain_ids']=torch.from_numpy(graph['chain_ids']).to( torch.int64)
                #data['dist_mat_index'], data['dist_mat_values'] =dense_to_sparse(torch.from_numpy(graph['dist_mat']))
                # to recover dense:
                #ds=to_dense_adj(edge_index=data['dist_mat_index'] , edge_attr=data['dist_mat_values'] ).squeeze(0)
                #torch.all(torch.eq(ds,dist_mat))
                return data
##########################################################################


class residue_granulatiy_InMemoryDataset(InMemoryDataset):
    def __init__(self, df=None, df_file=None,
                 files_folder='.',
                 pdbs_col='pdb_id',
                 properties= None ,
                 root='temp_datasets',
                 pdb_file_name='rec_h_opt.pdb',
                 pdb2graph=protein_pdb2graph,
                 transform=None,
                 pre_transform=None, 
                 force_reload=True):
        self.df_file = df_file
        self.df = df
        if self.df_file: 
            self.df = pd.read_csv(self.df_file)
            df.reset_index(drop=True, inplace=True)
        assert isinstance(self.df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        self.files_folder=files_folder
        self.pdb_file_name=pdb_file_name
        assert pdbs_col in self.df.columns , f'{pdbs_col} is not in the columns of {self.df}'
        self.pdbs_col= pdbs_col
        self.properties=properties
        if self.properties:
            for pro in self.properties: assert pro in self.df.columns , f'{pro} is not in the columns of df'
        #self.root = root
        self.root= osp.join(root)
        self.pdb2graph = pdb2graph
        super().__init__(root=self.root,  transform = transform, pre_transform=pre_transform)
        """
       
        """
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return df_file

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        pdbs_list = self.df[self.pdbs_col]
            
        print('Converting mol strings into graphs...')
        data_list = []
        for i in tqdm(range(len(pdbs_list))):
            
                #try:            
                pdb = pdbs_list[i]
                pdb_file=f'{self.files_folder}/{pdb}/{self.pdb_file_name}'
                #graph = self.pdb2graph(pdb_file)
                rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False)
                if rdkit_prot is None:
                    rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False,sanitize=False, proximityBonding=True)
                    Chem.GetSymmSSSR(rdkit_prot)
                protein_mol = Molecule.from_rdkit(rdkit_prot)
                data= protein_mol2pyg(protein_mol)
                data.pdb=pdb
                if self.properties:
                    for pro in self.properties: data[pro]=torch.Tensor([self.df.iloc[i][pro]])#.to(torch.float)          
                ############
                if self.pre_transform: data=self.pre_transform(data)
                data_list.append(data)
                #except:print('escape the following file which can not be convert into a graph', sdf_file)

        #if self.pre_transform is not None:
        #    print('running  pre_transformation .. ')
        #    data_list = [self.pre_transform(data) for data in data_list]
            
        print('Saving...')
        #data, slices = self.collate(data_list)
        #torch.save((data, slices), self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])

from torch_geometric.data import Dataset     


def process_pdb(pdb_tuple):
    i,pdb,pdb_file,properties,values,pre_transform,processed_dir=pdb_tuple 

    #check for existing graph data
    filenm=osp.join(processed_dir, f'geometric_data_processed_{pdb}.pt')
    if osp.exists(filenm):
        return f'{filenm} exists'
    else:
        #graph = self.pdb2graph(pdb_file)
        try:
            rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False)
            if rdkit_prot is None:
                print('None prot')
                rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False,sanitize=False, proximityBonding=True)
                Chem.GetSymmSSSR(rdkit_prot)
                
            protein_mol = Molecule.from_rdkit(rdkit_prot)
            
            data= protein_mol2pyg(protein_mol)
            data.pdb=pdb
            if properties:
                for pro,v in zip(properties,values): data[pro]=torch.Tensor(v)#.to(torch.float)
            ############
            if pre_transform: data=pre_transform(data)

            torch.save(data, osp.join(processed_dir, filenm))
            return f'{filenm} created'
        except Exception as e:
            print(f'{filenm} failed: {e}')
            return f'{filenm} failed'
    
    
        
class ResidueGranularityDataset(Dataset):
    def __init__(self, df=None, df_file=None,
                 pdbs_col='pdb_id',
                 pdbs_file_col=None,
                 properties= None ,
                 root='temp_datasets',
                 pdb2graph=protein_pdb2graph,
                 transform=None,
                 pre_transform=None, 
                 force_reload=True,
                 n_cores=4,
                 chunksize=10):
        self.df_file = df_file
        self.df = df
        if self.df_file: 
            self.df = pd.read_csv(self.df_file)
            df.reset_index(drop=True, inplace=True)
        assert isinstance(self.df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        assert pdbs_col in self.df.columns , f'{pdbs_col} is not in the columns of {self.df}'
        assert pdbs_file_col in self.df.columns , f'{pdbs_file_col} is not in the columns of {self.df}'
        self.pdbs_col= pdbs_col
        self.pdbs_file_col=pdbs_file_col
        self.properties=properties
        if self.properties:
            for pro in self.properties: assert pro in self.df.columns , f'{pro} is not in the columns of df'
        #self.root = root
        self.root= osp.join(root)
        self.pdb2graph = pdb2graph
        self.n_cores=n_cores
        self.chunksize=chunksize
        super().__init__(root=self.root,  transform = transform, pre_transform=pre_transform)
        """
       
        """
        #self.data, self.slices = torch.load(self.processed_paths[0])
        #self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return df_file

    @property
    def processed_file_names(self):
        return [f'geometric_data_processed_{pdb}.pt' for idx,pdb in enumerate(self.df[self.pdbs_col])]

    def process(self):
        pdbs_list = self.df[self.pdbs_col]
        pdbs_files= self.df[self.pdbs_file_col]
        assert len(pdbs_files)==len(pdbs_list)
        
        if self.properties:
            tuple_list=[ (i,pdbs_list[i],pdbs_files[i],self.properties,[ [self.df.iloc[i][pro]] for pro in self.properties], self.pre_transform, self.processed_dir) for i in range(len(pdbs_list))]
        else:
            tuple_list=[ (i,pdbs_list[i],pdbs_files[i],self.properties,[], self.pre_transform, self.processed_dir) for i in range(len(pdbs_list))]
            
        print('Converting mol pdbs into graphs...')
        
        message=[]
        #with Pool(self.n_cores) as p:
        #    res=p.imap_unordered(process_pdb, tuple_list,chunksize=self.chunksize)
        #    for out in res:
        #        message.append(out)
        for t in tqdm(tuple_list):
            message.append(process_pdb(t))
        
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        pdb=self.df[self.pdbs_col][idx]
        data = torch.load(osp.join(self.processed_dir,  f'geometric_data_processed_{pdb}.pt'),weights_only=False)
        return data
    
    
##############################################
from  automol.structurefeatures.GradFormer.utils_3d.get_interactions import interactions_2graph
def interactions_2graph_2_pyg(ligand_mol, protein_mol, return_PYG=False):
    data =dict()
    graph=interactions_2graph(ligand_mol, protein_mol)
    data['interactions_edges_list'] = torch.from_numpy(graph['edges_list']).to(torch.int64)
    data['interactions_One_hot_encoding']=torch.from_numpy(graph['interactions_One_hot_encoding']).to( torch.int64)
    data['interactions_parameters']=torch.from_numpy(graph['interactions_parameters']).float()
    data['ligand_delta_sasa_Polar']=torch.from_numpy(graph['ligand_delta_sasa_Polar']).float()
    data['ligand_delta_sasa_Apolar']=torch.from_numpy(graph['ligand_delta_sasa_Apolar']).float()
    data['protein_delta_sasa_Polar']=torch.from_numpy(graph['protein_delta_sasa_Polar']).float()
    data['protein_delta_sasa_Apolar']=torch.from_numpy(graph['protein_delta_sasa_Apolar']).float()
    if  return_PYG:
        pygdata = Data()
        for key, value in data.items(): pygdata[key]=value
        return pygdata   
    return data

def get_interactions_graph_pyg_data_list(df,
                                         pdbs_col='pdb_id',
                                         properties= ['A','B'] ,
                                         pdb_file_name='rec_h_opt.pdb',
                                         sdf_file_name='cry_lig_opt_converted.sdf',
                                         files_folder='.',
                                        lig_pre_transform=None,
                                        pro_pre_transform=None,
                                        same_receptor_str_all_ligands=True):
    
    print('Converting mol strings into graphs...')
    df.reset_index(drop=True, inplace=True)
    pdbs_list = df[pdbs_col]
    data_dic = dict()
    for i in tqdm(range(len(pdbs_list))):
        #try:            
        pdb = pdbs_list[i]
        sdf_file=f'{files_folder}/{pdb}/{sdf_file_name}'
        ligand_mol= Chem.SDMolSupplier(sdf_file,removeHs=False)[0]
        ligand_mol=Molecule.from_rdkit(ligand_mol)
        pdb_file=f'{files_folder}/{pdb}/{pdb_file_name}'
        rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False)
        if rdkit_prot is None:
            rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False,sanitize=False, proximityBonding=True)
            Chem.GetSymmSSSR(rdkit_prot)
        protein_mol = Molecule.from_rdkit(rdkit_prot)
        inter_graph=interactions_2graph(ligand_mol, protein_mol)
        #interaction_data= interactions_2graph_2_pyg(ligand_mol, protein_mol)
        lig_data=mol2_pyg(ligand_mol)
        lig_data.pdb=pdb
        for pro in properties: lig_data[pro]=torch.Tensor([df.iloc[i][pro]])
        if lig_pre_transform: lig_data=lig_pre_transform(lig_data)
        #lig_data['interactions_edges_list'] = torch.from_numpy(inter_graph['edges_list']).to(torch.int64)
        lig_data['interactions_One_hot_encoding']=torch.from_numpy(inter_graph['interactions_One_hot_encoding']).to( torch.int64)
        lig_data['interactions_parameters']=torch.from_numpy(inter_graph['interactions_parameters']).float()
        lig_data['ligand_delta_sasa_Polar']=torch.from_numpy(inter_graph['ligand_delta_sasa_Polar']).float()
        lig_data['ligand_delta_sasa_Apolar']=torch.from_numpy(inter_graph['ligand_delta_sasa_Apolar']).float()
        lig_data['protein_delta_sasa_Polar']=torch.from_numpy(inter_graph['protein_delta_sasa_Polar']).float()
        lig_data['protein_delta_sasa_Apolar']=torch.from_numpy(inter_graph['protein_delta_sasa_Apolar']).float()
        if pdb not in data_dic:
            pro_data= protein_mol2pyg(protein_mol)
            pro_data.pdb=pdb
            if pro_pre_transform: pro_data=pro_pre_transform(pro_data)
            #data_dic[pdb]={'protein': [pro_data], 'ligands':[],'interactions':[]}
            data_dic[pdb]={'protein': [pro_data], 'ligands':[]}#,'interactions':[]}
        else:
            if same_receptor_str_all_ligands: 
                pass
            else:
                pro_data= protein_mol2pyg(protein_mol)
                pro_data.pdb=pdb
                if pro_pre_transform: pro_data=pro_pre_transform(pro_data)
                data_dic[pdb]['protein'].append(pro_data)  
        data_dic[pdb]['ligands'].append(lig_data)
        #data_dic[pdb]['interactions'].append(interaction_data)
        
    return data_dic 

def process_interaction(interaction_tuple):
    pdb,sdf_name, sdf_file,pdb_file=interaction_tuple
    
    if isinstance(sdf_file,str):
        ligand_mol= Chem.SDMolSupplier(sdf_file,removeHs=False)[0]
        ligand_mol=Molecule.from_rdkit(ligand_mol)
    else:
        ligand_mol=Molecule.from_rdkit(sdf_file,chain='unk')
        

    rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if rdkit_prot is None:
        rdkit_prot = Chem.MolFromPDBFile(pdb_file, removeHs=False,sanitize=False, proximityBonding=True)
        Chem.GetSymmSSSR(rdkit_prot)
    protein_mol = Molecule.from_rdkit(rdkit_prot)
    inter_graph=interactions_2graph(ligand_mol, protein_mol)
    ## pyg batch does not accept pairs from diffrent graphs
    out_inter=dict()
    interactions_pairs=torch.from_numpy(inter_graph['edges_list']).to(torch.int64)
    out_inter['id'] = f'iteraction_{pdb}_{sdf_name}'
    if len(interactions_pairs.shape)>1:
        out_inter['interactions_pairs_lig'] = interactions_pairs[0]
        out_inter['interactions_pairs_pro'] = interactions_pairs[1]
    else:
        print(f'Warning empty interaction: {sdf_file} with {pdb}')
        out_inter['interactions_pairs_lig'] = torch.from_numpy(np.ndarray([0])).to( torch.int64)
        out_inter['interactions_pairs_pro'] = torch.from_numpy(np.ndarray([0])).to( torch.int64)
    out_inter['interactions_One_hot_encoding']=torch.from_numpy(inter_graph['interactions_One_hot_encoding']).to( torch.int64)
    out_inter['interactions_parameters']=torch.from_numpy(inter_graph['interactions_parameters']).float()
    out_inter['ligand_delta_sasa_Polar']=torch.from_numpy(inter_graph['ligand_delta_sasa_Polar']).float()
    out_inter['ligand_delta_sasa_Apolar']=torch.from_numpy(inter_graph['ligand_delta_sasa_Apolar']).float()
    out_inter['protein_delta_sasa_Polar']=torch.from_numpy(inter_graph['protein_delta_sasa_Polar']).float()
    out_inter['protein_delta_sasa_Apolar']=torch.from_numpy(inter_graph['protein_delta_sasa_Apolar']).float()
    return out_inter

class InteractionDataset():
    def __init__(self,df,
                     pdbs_col='pdb_id',
                     pdbs_file_col=None,
                     ligands_file_col=None,
                     properties= ['A','B'] ,
                     n_cores=4,
                     chunksize=100):
        self.df = df
        
        assert pdbs_col in self.df.columns , f'{pdbs_col} is not in the columns of {self.df}'
        assert pdbs_file_col in self.df.columns , f'{pdbs_file_col} is not in the columns of {self.df}'
        assert ligands_file_col in self.df.columns , f'{ligands_file_col} is not in the columns of {self.df}'
        self.pdbs_col = pdbs_col
        self.pdbs_file_col = pdbs_file_col
        self.ligands_file_col = ligands_file_col
        self.properties = properties
        self.n_cores=n_cores
        self.chunksize=chunksize
        
        interaction_tuples=[]
        for pdb,pdb_file,sdf_file in zip(self.df[self.pdbs_col],self.df[self.pdbs_file_col],self.df[self.ligands_file_col]):
            
            if sdf_file.endswith('cry_lig_opt_converted.sdf'):
                sdf_name='reference'
            else:
                try:
                    sdf_name=sdf_file.rsplit('/',1)[1].split('_')[2]
                except:
                    sdf_name=sdf_file.rsplit('/',1)[1]
            
            interaction_tuples.append((pdb,sdf_name,sdf_file,pdb_file))
            
        self.interactions=[]
        print('Computing ligand/protein interactions...')
        #with Pool(self.n_cores) as p:
        #    res=p.imap(process_interaction, interaction_tuples,chunksize=self.chunksize)
        #    for out in res:
        #        self.interactions.append(out)
                
        
        for t in tqdm(interaction_tuples):
            self.interactions.append(process_interaction(t))
            
        super(InteractionDataset, self).__init__()
        
 
    def __getitem__(self, indx):
        return self.interactions[indx]
 
    def __len__(self):
        return len(self.df)
    
class InteractionDataset_SDFfile():
    def __init__(self,
                     sdf_file='ligands.sdf',
                     pdb_file_folder='.',
                     pdb_key='pdb',
                     pdb_file_end='_split_receptor1.pdb',
                     properties= ['A','B'] ,
                     n_cores=4,
                     chunksize=100):
        
        self.sdf_file = sdf_file
        self.pdb_file_folder=pdb_file_folder
        self.pdb_key = pdb_key
        self.properties = properties
        self.n_cores=n_cores
        self.chunksize=chunksize
        
        interaction_tuples=[]
        for mol in tqdm(Chem.SDMolSupplier(sdf_file,removeHs=False)):
            pdb_nm=mol.GetProp(self.pdb_key)
            jnj_id=retrieve_prop_from_mol(mol, guesses=['JNJNUMBER','JNJ','CHEMBLID'], start='JNJ',remove_q=False)
            if sdf_file.endswith('cry_lig_opt_converted.sdf'):
                sdf_name='reference'
            else:
                try:
                    sdf_name=sdf_file.rsplit('/',1)[1].split('_')[2]
                except:
                    sdf_name=sdf_file.rsplit('/',1)[1]
            
            interaction_tuples.append((pdb_nm,f'JNJ_{jnj_id}',mol,f'{self.pdb_file_folder}{pdb_nm}{pdb_file_end}'))

        self.interactions=[]
        print('Computing ligand/protein interactions...')
        #with Pool(self.n_cores) as p:
        #    res=p.imap(process_interaction, interaction_tuples,chunksize=self.chunksize)
        #    for out in res:
        #        self.interactions.append(out)
                
        
        for t in tqdm(interaction_tuples):
            self.interactions.append(process_interaction(t))
            
        super(InteractionDataset_SDFfile, self).__init__()
        
 
    def __getitem__(self, indx):
        return self.interactions[indx]
 
    def __len__(self):
        return len(self.df)
    
def fn_interaction_pad(data_list):
    """
    convert the sparse mat to dense for the MHA and pad them t othe same size
    """
    batched_dict={}
    for data in data_list:
        #data=data_adj2dense(data, adj_mat_2dense)
        for ki, k in data.items():
            if ki not in batched_dict:
                batched_dict[ki]=[]
            batched_dict[ki].append(k)
    return {ki: ( k if isinstance(k[0],str) else pad_sequence(k, batch_first=True, padding_value=-510) ) for ki,k in batched_dict.items()}


#########################################################################################
import random

class CumulativeDataset():
    def __init__(self,index_coor,
                     protein_dataset=None,
                     ligand_dataset=None,
                     interaction_dataset=None
                ):
        self.index_coor = index_coor
        self.protein_dataset = protein_dataset
        self.ligand_dataset = ligand_dataset
        self.interaction_dataset = interaction_dataset
        super(CumulativeDataset, self).__init__()
        
 
    def __getitem__(self, indx):
        out_dict={}
        pro_indx,lig_list=self.index_coor[indx]
        out_dict['protein']=self.protein_dataset[pro_indx]
        out_dict['protein_indices']=torch.tensor(pro_indx)
        out_dict['ligands']=[]
        out_dict['interactions']=[]
        out_dict['indices']=[]
        for j in lig_list:
            out_dict['ligands'].append(self.ligand_dataset[j])
            out_dict['interactions'].append(self.interaction_dataset[j])
            out_dict['indices'].append(torch.tensor([pro_indx,j]))
            
        return out_dict
    
    def test_i_indices(self,i,seed=None):
        if seed is not None:
            random.seed(seed)
        proteins_idx=[random.randint(0, len(self.index_coor)-1) for i in range(i)]
        for indx in proteins_idx:
            pro_indx,lig_list=self.index_coor[indx]
            prot=self.protein_dataset[pro_indx]
            for j in lig_list:
                ligand=self.ligand_dataset[j]
                interact=self.interaction_dataset[j]
                assert prot.pdb==ligand.pdb
                assert prot.pdb==interact['id'].split('_')[1]
                
                
    def generate_weights(self,prop='',eps=1e-3,boundaries=torch.tensor([0,4,5,5.5,6,6.5,7,7.5,8,9,10,12,float('inf')])):
        aff_values= self.ligand_dataset.df[prop]
        values=[]
        for prot_indx,ligandidx in self.index_coor:
            for i in ligandidx:
                values.append(aff_values[i])
        bucket_values=torch.bucketize(torch.tensor(values), boundaries)
        bucket_weights=1./ (torch.bincount(bucket_values)+eps)
        weights=[]
        idx=0
        for prot_indx,ligandidx in self.index_coor:
            index_weight=0
            for i in ligandidx:
                index_weight+=bucket_weights[bucket_values[idx]]
                idx+=1
            weights.append(index_weight)
        return torch.tensor(weights)
 
    def __len__(self):
        #nb of proteins
        return len(self.index_coor)

def generate_fixed_coor(protein_ind,max_ligand_pairs=3):
    splitted_pairs=[]
    for p,pl in protein_ind:
        random.shuffle(pl)
        split_list=[pl[i:i + max_ligand_pairs] for i in range(0, len(pl), max_ligand_pairs)]
        for sp in split_list:
            splitted_pairs.append((p,sp))
    return splitted_pairs

def update_fixed_coor(splitted_pairs,max_ligand_pairs=3):
    dict_pt={}
    for pt in splitted_pairs:
        if pt[0] in dict_pt:
            dict_pt[pt[0]].extend(pt[1])
        else:
            dict_pt[pt[0]]=pt[1]

    protein_ind=[(key,item) for key,item in dict_pt.items()]
    splitted_idx=0
    for p,pl in protein_ind:
        random.shuffle(pl)
        split_list=[pl[i:i + max_ligand_pairs] for i in range(0, len(pl), max_ligand_pairs)]
        for sp in split_list:
            splitted_pairs[splitted_idx]=(p,sp)
            splitted_idx+=1
    return splitted_pairs

def generate_hierarchical_triplets(protein_ind,max_ligand_pairs=3):
    splitted_pairs=[]
    for p,pl in protein_ind:
        all_pairs=[(x, y) for x, y in itertools.product(pl, pl) if x != y] 
        random.shuffle(all_pairs)
        split_list=[all_pairs[i:i + max_ligand_pairs] for i in range(0, len(all_pairs), max_ligand_pairs)]
        for sp in split_list:
            splitted_pairs.append((p,sp))
    return splitted_pairs
            
    
class CumulativePairedDataset():
    def __init__(self,index_triplets,
                     protein_dataset=None,
                     ligand_dataset=None,
                     interaction_dataset=None
                ):
        self.index_triplets = index_triplets
        self.protein_dataset = protein_dataset
        self.ligand_dataset = ligand_dataset
        self.interaction_dataset = interaction_dataset
        super(CumulativePairedDataset, self).__init__()
        
 
    def __getitem__(self, indx):
        out_dict={}
        prot_indx,ligand_tuples=self.index_triplets[indx]
        out_dict['protein']=self.protein_dataset[prot_indx]
        out_dict['protein_indices']=torch.tensor(prot_indx)
        out_dict['ligands_one']=[]
        out_dict['interactions_one']=[]
        out_dict['indices_one']=[]
        out_dict['ligands_two']=[]
        out_dict['interactions_two']=[]
        out_dict['indices_two']=[]
        for (i,j) in ligand_tuples:
            out_dict['ligands_one'].append(self.ligand_dataset[i])
            out_dict['ligands_two'].append(self.ligand_dataset[j])
            out_dict['interactions_one'].append(self.interaction_dataset[i])
            out_dict['interactions_two'].append(self.interaction_dataset[j])
            out_dict['indices_one'].append(torch.tensor([prot_indx,i]))
            out_dict['indices_two'].append(torch.tensor([prot_indx,j]))
            
        return out_dict
    
    def test_i_indices(self,i,seed=None):
        if seed is not None:
            random.seed(seed)
        item_idx=[random.randint(0, len(self.index_triplets)-1) for i in range(i)]
        for indx in item_idx:
            prot_idx,pair_idx=self.index_triplets[indx]
            prot=self.protein_dataset[prot_idx]
            for i,j in pair_idx:
                ligand_1=self.ligand_dataset[i]
                interact_1=self.interaction_dataset[i]
                ligand=self.ligand_dataset[j]
                interact=self.interaction_dataset[j]
                assert prot.pdb==ligand.pdb
                assert prot.pdb==interact['id'].split('_')[1]
                assert prot.pdb==ligand_1.pdb
                assert prot.pdb==interact_1['id'].split('_')[1]
                
    def generate_weights(self,prop='',eps=1e-3,boundaries=torch.tensor([float('-inf'),-3,-2,-1.2,-0.7,-0.3,0.3,0.7,1.2,3,float('inf')])):
        aff_values= self.ligand_dataset.df[prop]
        values=[]
        for prot_indx,ligand_tuples in self.index_triplets:
            for i,j in ligand_tuples:
                values.append(aff_values[i]-aff_values[j])
        bucket_values=torch.bucketize(torch.tensor(values), boundaries)
        bucket_weights=1./ (torch.bincount(bucket_values)+eps)
        
        weights=[]
        idx=0
        for prot_indx,ligand_tuples in self.index_triplets:
            index_weight=0
            for i,j in ligand_tuples:
                index_weight+=bucket_weights[bucket_values[idx]]
                idx+=1
            weights.append(index_weight)
        return torch.tensor(weights)
            
 
    def __len__(self):
        #nb of proteins
        return len(self.index_triplets)

def collate_tensor_fn(batch):
    return torch.stack(batch, 0)    

def alldata_collate_fn(data_list,list_keys,params):
    batched_dict={}
    for data in data_list:
        for ki, k in data.items():
            if ki not in batched_dict:
                batched_dict[ki]=[]
            if ki in list_keys:
                for kv in k:
                    batched_dict[ki].append(kv)   
            else:
                batched_dict[ki].append(k)
    out_dict={}
    for key,val in params.items():
        if 'params' in val:
            out_dict[key]=val['fn'](batched_dict[key],val['params'])
        else:
            out_dict[key]=val['fn'](batched_dict[key])
            
    return out_dict


#################################################
def data_sparse2dense(data, adj_mat_2dense=[('dist_mat_index','dist_mat_values' ),
                                        ('topological_dist_mat_index','topological_dist_mat_values' ),
                                       ]):
    for ki, k in adj_mat_2dense:
        ds=to_dense_adj(edge_index=data[ki] , edge_attr=data[k] ).squeeze(0)
        data.remove_tensor(k)
        data.remove_tensor(ki)
        data[k]=ds
    return data


def fn_sparse2dense(data_list,soarse_mat_2dense=[('dist_mat_index','dist_mat_values' ),
                                        ('topological_dist_mat_index','topological_dist_mat_values' ),
                                       ]):
    """
    convert the sparse mat to dense for the MHA and pad them t othe same size
    """
    max_num_nodes = max([data.num_nodes for data in data_list])
    nn_list=[]
    for data in data_list:
        #data=data_adj2dense(data, adj_mat_2dense)
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        for ki, k in soarse_mat_2dense:
            ds=to_dense_adj(edge_index=data[ki] , edge_attr=data[k] ).squeeze(0)
            data.remove_tensor(k)
            data.remove_tensor(ki)
            data[k] = torch.nn.functional.pad(ds, (0, pad_size, 0, pad_size), value=510)
        nn_list.append(data)
    batched_data = Batch.from_data_list(nn_list)
    return batched_data



###########################################
class Compose_and_cat(BaseTransform):
    r"""Composes several transforms and cat them together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """
    def __init__(self, transforms,
                 attr_name='pe'
                ):
        self.transforms = transforms
        self.attr_name=attr_name

    def forward(    self,    data ) :
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [self.tranform_one_data(d) for d in data]
            else:
                data = self.tranform_one_data(data)
        return data
    def tranform_one_data(self, data):
        for transform in self.transforms:
            data = transform(data)
        catted= [data[transform.attr_name] for  transform in self.transforms]  
        data[self.attr_name] = torch.cat(catted, dim=-1)
        return data

    def __repr__(self) -> str:
        args = [f'  {transform}' for transform in self.transforms]
        return 'cat_pe_{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))
################################


##########################
class online_gps_Dataset(torch.utils.data.Dataset):
    #def __init__(self, smiles, smiles2graph,pre_transform=None ):
    def __init__(self, df=None, df_file=None, smiles='smiles',properties= ['A','B'] , smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        super().__init__()
    
        self.df_file = df_file
        self.df = df
        if self.df_file: 
            self.df = pd.read_csv(self.df_file)
        df.reset_index(drop=True, inplace=True)
        assert isinstance(self.df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        assert smiles in self.df.columns , f'{smiles} is not in the columns of {self.df}'
        self.smiles= smiles
        self.properties=properties
        for pro in self.properties:
            assert pro in self.df.columns , f'{pro} is not in the columns of df'
        self.smiles2graph = smiles2graph 
        self.pre_transform=pre_transform
    def __getitem__(self, index):
        s=self.df.iloc[index][self.smiles]
        #print(index, s)
        data = Data()
        graph = self.smiles2graph(s)
        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        data['sph']=sp
        if self.pre_transform: data=self.pre_transform(data)
        for pro in self.properties: data[pro]=self.df.iloc[index][pro]
        return data

    def __len__(self):
        return len(self.df)
#######################
#this data set is used by the model for infrence 
##########################
class infer_Dataset(torch.utils.data.Dataset):
    def __init__(self, smiles, smiles2graph,pre_transform=None ):
        super().__init__()
        self.smiles = smiles
        self.smiles2graph=smiles2graph
        self.pre_transform=pre_transform
        
    def __getitem__(self, index):
        s=self.smiles[index]
        data = Data()
        graph = self.smiles2graph(s)
        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        data['sph']=sp
        if self.pre_transform: data=self.pre_transform(data)
        return data

    def __len__(self):
        return len(self.smiles)
def prepare_batch_w_sph(data_list):
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data
###############################

###########################################################################
def fn(data_list):
    max_num_nodes = max([data.sph.shape[0] for data in data_list])
    for data in data_list:
        num_nodes = data.num_nodes
        pad_size = max_num_nodes - num_nodes
        data.sph = torch.nn.functional.pad(data.sph, (0, pad_size, 0, pad_size), value=510)
    batched_data = Batch.from_data_list(data_list)
    return batched_data

#######################
class NewDataset_w_sph(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.dataset = original_dataset
        self.sph = []
        self.get_sph_all()
        assert len(self.sph)== len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        data['sph'] = self.sph[index]
        return data

    def __len__(self):
        return len(self.dataset)
    def get_sph_all(self):
        self.sph = []
        file = osp.join('/'.join(self.dataset.processed_paths[0].split('/')[:-1]), 'sph.pkl')
        if not os.path.exists(file):
            print('pre-process sph start!')
            progress_bar = tqdm(desc='pre-processing Data', total=len(self.dataset), ncols=40)
            for i in range(len(self.dataset)):
                self.process(i)
                progress_bar.update(1)
            progress_bar.close()
            pickle.dump(self.sph, open(file, 'wb'))
            print('pre-process sph done!')
        else:
            self.sph = pickle.load(open(file, 'rb'))
            print('load sph done!')

    def process(self, index):
        data = self.dataset[index]
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        self.sph.append(sp)
        """
        self.sph.append(adj_pinv(data, topk=10)) # ectd
        mat = cosine_similarity(data.x, data.x)
        mat = 1 / mat
        self.sph.append(mat) # cosine similarity
        """


##########################################################################
class custom_InMemoryDataset(InMemoryDataset):
    def __init__(self, df=None, df_file=None, smiles='smiles',properties= ['A','B'] ,root='temp_datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None, force_reload=True):
        self.df_file = df_file
        self.df = df
        if self.df_file: 
            self.df = pd.read_csv(self.df_file)
            df.reset_index(drop=True, inplace=True)
        assert isinstance(self.df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        assert smiles in self.df.columns , f'{smiles} is not in the columns of {self.df}'
        self.smiles= smiles
        self.properties=properties
        for pro in self.properties:
            assert pro in self.df.columns , f'{pro} is not in the columns of df'
    
        #self.root = root
        self.root= osp.join(root)
        self.smiles2graph = smiles2graph
        super().__init__(root=self.root,  transform = transform, pre_transform=pre_transform)
        """
       
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        dataset = custom_InMemoryDataset()
        print(dataset)
        print(dataset.data.edge_index)
        print(dataset.data.edge_index.shape)
        print(dataset.data.x.shape)
        print(dataset[100])
        print(dataset.get_idx_split())
        """
        #if self.df_file: df = pd.read_csv(self.df_file) 
        #assert isinstance(df,pd.core.frame.DataFrame), 'provide pandas dataframe'
        
        #self.folder = osp.join(root, 'peptides-structural')
        """
        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)
        """
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return df_file

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        smiles_list = self.df[self.smiles]

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            try:            
                data = Data()
                smiles = smiles_list[i]
                graph = self.smiles2graph(smiles)
                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])
    
                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(
                    torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                    torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                #data.y = torch.Tensor([y])
                #y = data_df.iloc[i][properties]
                for pro in self.properties: data[pro]=self.df.iloc[i][pro]
                ########
                """
                N = data.x.shape[0]
                adj = torch.zeros([N, N])
                adj[data.edge_index[0, :], data.edge_index[1, :]] = True
                data.sph, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
                """
                ############
                data=self.pre_transform(data)
                data_list.append(data)
            except:
                print('escape the following smiles which can not be convert into a graph', smiles)

        #if self.pre_transform is not None:
        #    print('running  pre_transformation .. ')
        #    data_list = [self.pre_transform(data) for data in data_list]
            
        print('Saving...')
        #data, slices = self.collate(data_list)
        #torch.save((data, slices), self.processed_paths[0])
        self.save(data_list, self.processed_paths[0])


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


###########################
class NewDataset(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.dataset = original_dataset
        self.sph = []

    def __getitem__(self, index):
        data = self.dataset[index]
        data['sph'] = self.sph[index]
        return data

    def __len__(self):
        return len(self.dataset)

    def process(self, index):
        data = self.dataset[index]
        N = data.x.shape[0]
        adj = torch.zeros([N, N])
        adj[data.edge_index[0, :], data.edge_index[1, :]] = True
        sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
        self.sph.append(sp)
        """
        self.sph.append(adj_pinv(data, topk=10)) # ectd
        mat = cosine_similarity(data.x, data.x)
        mat = 1 / mat
        self.sph.append(mat) # cosine similarity
        """


