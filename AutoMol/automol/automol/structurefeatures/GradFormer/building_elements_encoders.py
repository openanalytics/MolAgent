import torch
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential, Parameter, LeakyReLU, BatchNorm1d
import numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math, copy ,os ,inspect

#from dataset_utils.dataset import infer_Dataset, prepare_batch_w_sph
from torch.utils.data import  DataLoader, Dataset
from torch import optim
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, GINConv, GINEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
######
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
#from dataset_utils.utils import process_hop
from automol.structurefeatures.GradFormer.utils_3d.protein_features_util import get_res_feature_dims, get_res_res_bond_feature_dims, get_Noncovalent_bond_feature_dims

from automol.structurefeatures.GradFormer.utils_3d.ligand_features_util import get_atom_feature_dims, get_bond_feature_dims 
from automol.structurefeatures.GradFormer.utils_3d.ligand_features_util import mol2graph , sdf2graph
from automol.structurefeatures.GradFormer.grad_conv import GPSConv, GatedGCNLayer 
#######

###############################################################################
#############  encoders atoms and atomistic based interactions  ###############
###############################################################################

##############################################
class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, add_sasas=False):
        super(AtomEncoder, self).__init__()
        self.hidden_dim=emb_dim
        self.atom_embedding_list = torch.nn.ModuleList()
        full_atom_feature_dims = get_atom_feature_dims()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        self.sasa_emb=None
        if add_sasas:
            self.sasa_emb=Linear(4, emb_dim)  
    def forward(self, x, sasa_atom_wise=None):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])
        if self.sasa_emb:
            x_embedding +=self.sasa_emb(sasa_atom_wise)
        return x_embedding


##############################################
class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        full_bond_feature_dims = get_bond_feature_dims()
        self.hidden_dim=emb_dim
        self.bond_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding
##############################################
# encode atomistic mol 
##############################################
class mol_atomstic_encoder(torch.nn.Module):
    def __init__(self, hidden_dim=100, sasa_hidden_dim=10,  
                 mpnn='GINE',
                 pe_origin_dim=20,pe_dim=20 ,
                 gamma=0.5, slope=0.0, hops_list=None,
                 nhead=4,n_hop=5,num_layers=10, 
                 dropout=0.1, drop_prob=0.0, attn_dropout=0.1
                ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.pe_lin = Linear(pe_origin_dim, pe_dim)
        self.sasa_enc=Linear(4, sasa_hidden_dim)
        self.atm_enc=AtomEncoder( emb_dim=hidden_dim- pe_dim -sasa_hidden_dim, add_sasas=False)
        self.bond_enc=BondEncoder( emb_dim=hidden_dim)
        self.gamma = gamma
        self.slope = slope
        self.mpnn = mpnn
        if hops_list:
            assert len(hops_list)== nhead
            self.hop = Parameter(torch.tensor(hops_list).reshape([nhead,1,1]))
        else:
            self.hop = Parameter(torch.full((nhead, 1, 1), float(n_hop)))
        self.convs = ModuleList()
        if mpnn == 'GIN':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GINE':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINEConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GCN':
            #Lap_pe = False
            for _ in range(num_layers):
                conv = GPSConv(hidden_dim, GatedGCNLayer(channels, channels, 0, True,
                                                            equivstable_pe=Lap_pe),
                               heads=nhead, dropout=dropout, attn_dropout=attn_dropout,
                               drop_prob=drop_prob)
                self.convs.append(conv)

    def forward(self, x, sasa, pe, edge_index, edge_attr, batch, sph, return_dense=True):
        #print('x=',x.size(),'edge_index',edge_index.size(),'pe',pe.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # process node
        x = torch.cat((self.atm_enc(x), self.sasa_enc(torch.nan_to_num(sasa)),self.pe_lin(pe)), 1)
        if torch.isnan(x).any():
            print(f'1 {x}, {sasa}, {pe}')
        #print('x=',x.size(),'edge_index',edge_index.size(),'pe',pe.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # process edge
        edge_attr = self.bond_enc(edge_attr)
        if torch.isnan(edge_attr).any():
            print(f'1 {edge_attr}')

        # get the sph
        sph = process_hop(sph, self.gamma, self.hop, self.slope)
        if torch.isnan(sph).any():
            print(f'2 {sph}')
        #print('x=',x.size(),'edge_index',edge_index.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # Conv & MPNN
        if self.mpnn == 'GIN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch)
        elif self.mpnn == 'GCN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr, pe=pe)
        elif self.mpnn == 'GINE':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr)
        #print("x before pooling=", x.size())
        if torch.isnan(x).any():
            print(f'3 {x}')
        if return_dense:
            h, nomask = to_dense_batch(x, batch)
            return h, nomask
            #h.masked_fill(~nomask.unsqueeze(-1), float('-inf'))
            #from torch_geometric.utils import mask_select, mask_to_index, select

        else:
            return x
        
class mol_residue_atomstic_encoder(torch.nn.Module):
    def __init__(self, hidden_dim=100, sasa_hidden_dim=10,  
                 mpnn='GINE',
                 pe_origin_dim=20,pe_dim=20 ,
                 gamma=0.5, slope=0.0, hops_list=None,
                 nhead=4,n_hop=5,num_layers=10, 
                 dropout=0.1, drop_prob=0.0, attn_dropout=0.1
                ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.pe_lin = Linear(pe_origin_dim, pe_dim)
        self.sasa_enc=Linear(4, sasa_hidden_dim)
        self.atm_enc=AtomEncoder( emb_dim=hidden_dim- pe_dim -sasa_hidden_dim, add_sasas=False)
        self.bond_enc=BondEncoder( emb_dim=hidden_dim)
        self.gamma = gamma
        self.slope = slope
        self.mpnn = mpnn
        if hops_list:
            assert len(hops_list)== nhead
            self.hop = Parameter(torch.tensor(hops_list).reshape([nhead,1,1]))
        else:
            self.hop = Parameter(torch.full((nhead, 1, 1), float(n_hop)))
        self.convs = ModuleList()
        if mpnn == 'GIN':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GINE':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINEConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GCN':
            #Lap_pe = False
            for _ in range(num_layers):
                conv = GPSConv(hidden_dim, GatedGCNLayer(channels, channels, 0, True,
                                                            equivstable_pe=Lap_pe),
                               heads=nhead, dropout=dropout, attn_dropout=attn_dropout,
                               drop_prob=drop_prob)
                self.convs.append(conv)

    def forward(self, x,res_enc, sasa, pe, edge_index, edge_attr, batch, sph, return_dense=True):
        #print('x=',x.size(),'edge_index',edge_index.size(),'pe',pe.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # process node
        x = torch.cat((self.atm_enc(x), self.sasa_enc(torch.nan_to_num(sasa)),self.pe_lin(pe)), 1)
        x += res_enc
        if torch.isnan(x).any():
            print(f'1 {x}, {sasa}, {pe}')
        #print('x=',x.size(),'edge_index',edge_index.size(),'pe',pe.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # process edge
        edge_attr = self.bond_enc(edge_attr)
        if torch.isnan(edge_attr).any():
            print(f'1 {edge_attr}')

        # get the sph
        sph = process_hop(sph, self.gamma, self.hop, self.slope)
        if torch.isnan(sph).any():
            print(f'2 {sph}')
        #print('x=',x.size(),'edge_index',edge_index.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # Conv & MPNN
        if self.mpnn == 'GIN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch)
        elif self.mpnn == 'GCN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr, pe=pe)
        elif self.mpnn == 'GINE':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr)
        #print("x before pooling=", x.size())
        if torch.isnan(x).any():
            print(f'3 {x}')
        if return_dense:
            h, nomask = to_dense_batch(x, batch)
            return h, nomask
            #h.masked_fill(~nomask.unsqueeze(-1), float('-inf'))
            #from torch_geometric.utils import mask_select, mask_to_index, select

        else:
            return x
        
class mol_light_atomstic_encoder(torch.nn.Module):
    def __init__(self, hidden_dim=100,
                 edge_dim=80,
                 mpnn='GINE',
                 pe_origin_dim=20,pe_dim=20 ,sasa_in=4, sasa_hidden_dim=12,
                 gamma=0.5, slope=0.0, hops_list=None,
                 nhead=4,n_hop=5,num_layers=10, 
                 dropout=0.1, drop_prob=0.0, attn_dropout=0.1
                ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.edge_dim=edge_dim
        self.sasa_enc=Linear(sasa_in, sasa_hidden_dim)
        self.pe_lin = Linear(pe_origin_dim, pe_dim)
        self.atm_enc=AtomEncoder( emb_dim=hidden_dim- pe_dim -sasa_hidden_dim, add_sasas=False)
        self.bond_enc=BondEncoder( emb_dim=hidden_dim)
        self.gamma = gamma
        self.slope = slope
        self.mpnn = mpnn
        if hops_list:
            assert len(hops_list)== nhead
            self.hop = Parameter(torch.tensor(hops_list).reshape([nhead,1,1]))
        else:
            self.hop = Parameter(torch.full((nhead, 1, 1), float(n_hop)))
        self.convs = ModuleList()
        if mpnn == 'GIN':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GINE':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINEConv(nn, edge_dim=self.edge_dim), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GCN':
            #Lap_pe = False
            for _ in range(num_layers):
                conv = GPSConv(hidden_dim, GatedGCNLayer(channels, channels, 0, True,
                                                            equivstable_pe=Lap_pe),
                               heads=nhead, dropout=dropout, attn_dropout=attn_dropout,
                               drop_prob=drop_prob)
                self.convs.append(conv)

    def forward(self, x, sasa, pe, edge_index, edge_attr, batch, sph, return_dense=True):
        #print('x=',x.size(),'edge_index',edge_index.size(),'pe',pe.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # process node
        #x = torch.cat((self.atm_enc(x), self.pe_lin(pe)), 1)
        x = torch.cat((x, self.pe_lin(pe),self.sasa_enc(torch.nan_to_num(sasa))), 1)
        #print('x=',x.size(),'edge_index',edge_index.size(),'pe',pe.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # process edge
        #edge_attr = self.bond_enc(edge_attr)
        edge_attr = edge_attr

        # get the sph
        sph = process_hop(sph, self.gamma, self.hop, self.slope)
        #print('x=',x.size(),'edge_index',edge_index.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        # Conv & MPNN
        if self.mpnn == 'GIN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch)
        elif self.mpnn == 'GCN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr, pe=pe)
        elif self.mpnn == 'GINE':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=edge_attr)
        #print("x before pooling=", x.size())
        if return_dense:
            h, nomask = to_dense_batch(x, batch)
            return h, nomask
            #h.masked_fill(~nomask.unsqueeze(-1), float('-inf'))
            #from torch_geometric.utils import mask_select, mask_to_index, select

        else:
            return x


#######################################################################################
################### encoders for residues and internal coor of proteins ###############
#######################################################################################

##############################################
class ResidueEncoder(torch.nn.Module):
    def __init__(self, emb_dim,add_sasas=False):
        super(ResidueEncoder, self).__init__()
        self.hidden_dim=emb_dim
        self.res_embedding_list = torch.nn.ModuleList()
        full_res_feature_dims = get_res_feature_dims()
        for i, dim in enumerate(full_res_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.res_embedding_list.append(emb)
        self.add_sasas=add_sasas
        if self.add_sasas:
            self.sasa_emb=Linear(4, emb_dim)

    def forward(self, x, sasa=None):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.res_embedding_list[i](x[:,i])
        if self.add_sasas:
            x_embedding +=self.sasa_emb(sasa)
        return x_embedding
###############################################
#######################################################################
from automol.structurefeatures.GradFormer.utils_3d.protein_features_util import possible_chain_IDs
class Residues_LearnedPositionalEmbedding(nn.Module):
    def __init__(self,  emb_dim=100,max_len_res_ids=10000):
        super().__init__()
        self.max_len_res_ids=max_len_res_ids
        self.residues_ids_position_embeddings= nn.Embedding(self.max_len_res_ids, emb_dim,padding_idx=-1)
        torch.nn.init.xavier_uniform_(self.residues_ids_position_embeddings.weight.data)
        self.max_len_chain_ids=len(possible_chain_IDs)
        self.chains_ids_position_embeddings= nn.Embedding(self.max_len_chain_ids, emb_dim, padding_idx=self.max_len_chain_ids-1)
        torch.nn.init.xavier_uniform_(self.chains_ids_position_embeddings.weight.data)
        
    def forward(self,residues_ids,chain_ids ):
        res_embIDs=self.residues_ids_position_embeddings(residues_ids)
        ch_embIDs=self.chains_ids_position_embeddings(chain_ids)
        return res_embIDs + ch_embIDs
##############
class dihedral_angles_Embedding_Encoder(torch.nn.Module):
    def __init__(self, emb_dim,max_nr_dihs=4, bin_interval=5):
        super(dihedral_angles_Embedding_Encoder, self).__init__()
        self.hidden_dim=emb_dim
        self.bins=torch.range(-180,180,bin_interval)
        self.padding= len(self.bins)
        embd_size=len(self.bins)+1 
        #print('self.padding',self.padding,embd_size)
        self.rotamers_emb= torch.nn.Embedding(embd_size, emb_dim, padding_idx=self.padding)
        torch.nn.init.xavier_uniform_(self.rotamers_emb.weight.data)
        self.linear=Linear(max_nr_dihs*emb_dim, emb_dim)
    def forward(self, rotamers):
        #device = rotamers.device
        mask=torch.isnan(rotamers).to(rotamers)
        position_ids=torch.bucketize(rotamers,self.bins.to(rotamers))
        position_ids[mask]=self.padding
        x_embedding =self.rotamers_emb(position_ids)
        ns=x_embedding.size()[:-2]
        x_embedding=x_embedding.view(*ns,-1)
        x_embedding=self.linear(x_embedding)
        #x_embedding=torch.sum(x_embedding, dim=1)
        return x_embedding#torch.nan_to_num(x_embedding, nan=0.0)

##############################################################################################
class res_res_BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(res_res_BondEncoder, self).__init__()
        self.hidden_dim=emb_dim
        full_bond_feature_dims = get_res_res_bond_feature_dims()
        self.bond_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
        return bond_embedding   

##############################################
"""
class res_res_Bakbone_dihs_Encoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(res_res_Bakbone_dihs_Encoder, self).__init__()
        self.res_res_BB_dihs_emb=Linear(2, emb_dim)
    def forward(self, res_res_BB_dihs ):
        x_embedding =self.res_res_BB_dihs_emb(res_res_BB_dihs )
        return torch.nan_to_num(x_embedding, nan=0.0)
        """
##############################################
class res_res_Non_covalent_BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(res_res_Non_covalent_BondEncoder, self).__init__()
        self.hidden_dim=emb_dim
        full_bond_feature_dims = get_Noncovalent_bond_feature_dims()
        self.bond_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
    def forward(self, noncovalent_edge_attr ):
        embedding = 0
        for i in range(noncovalent_edge_attr.shape[1]):
            embedding += self.bond_embedding_list[i](noncovalent_edge_attr[:,i])
        return embedding 


#############################################################################################
# encode residues 
#############################################################################################
class mol_residues_level_encoder(torch.nn.Module):
    def __init__(self, hidden_dim=100, sasa_hidden_dim=10,
                 emb_add=True, ## add the emb for pe and rotamers to (residues,sasa) and res_res_BondEncoder+ Bakbone_dihs_enc
                 rotamers_hidden_dim=20,   pe_dim=20 ,# ignore with emb_add
                 mpnn='GINE',
                 gamma=0.5, slope=0.0, hops_list=None,
                 nhead=4,n_hop=5,num_layers=10, 
                 dropout=0.1, drop_prob=0.0, attn_dropout=0.1
                ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.sasa_enc=Linear(4, sasa_hidden_dim)
        self.emb_add=emb_add
        if emb_add:
            self.pe_lin=Residues_LearnedPositionalEmbedding(emb_dim=hidden_dim)
            self.rotamers_enc= dihedral_angles_Embedding_Encoder( emb_dim=hidden_dim,max_nr_dihs=4)
            self.residue_enc=ResidueEncoder( emb_dim=hidden_dim-sasa_hidden_dim, add_sasas=False)
            ### bonded interaction
            self.res_res_covalent_bonds_enc=res_res_BondEncoder(emb_dim=hidden_dim)
            self.Bakbone_dihs_enc=dihedral_angles_Embedding_Encoder(emb_dim=hidden_dim,max_nr_dihs=2 )
        else:
            self.pe_lin=Residues_LearnedPositionalEmbedding(emb_dim=pe_dim)
            self.rotamers_enc= dihedral_angles_Embedding_Encoder( emb_dim=rotamers_hidden_dim)
            self.residue_enc=ResidueEncoder( emb_dim=hidden_dim- pe_dim-rotamers_hidden_dim -sasa_hidden_dim, add_sasas=False)
            ### bonded interaction
            res_res_Bond_dim=hidden_dim//2
            self.res_res_covalent_bonds_enc=res_res_BondEncoder(emb_dim=res_res_Bond_dim)
            self.Bakbone_dihs_enc=dihedral_angles_Embedding_Encoder(emb_dim=hidden_dim - res_res_Bond_dim)
        self.res_res_Non_covalent_bonds_enc=res_res_Non_covalent_BondEncoder(emb_dim=hidden_dim)
        self.gamma = gamma
        self.slope = slope
        self.mpnn = mpnn
        if hops_list:
            assert len(hops_list)== nhead
            self.hop = Parameter(torch.tensor(hops_list).reshape([nhead,1,1]))
        else:
            self.hop = Parameter(torch.full((nhead, 1, 1), float(n_hop)))
        self.convs = ModuleList()
        if mpnn == 'GIN':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GINE':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINEConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GCN':
            #Lap_pe = False
            for _ in range(num_layers):
                conv = GPSConv(hidden_dim, GatedGCNLayer(channels, channels, 0, True,
                                                            equivstable_pe=Lap_pe),
                               heads=nhead, dropout=dropout, attn_dropout=attn_dropout,
                               drop_prob=drop_prob)
                self.convs.append(conv)

    def forward(self, x, sasa_residue_wise , rotamers, residue_ids,chain_ids, edge_index,
                covalent_edge_index, covalent_edge_attr,res_res_BB_dihs,
                noncovalent_edge_index, noncovalent_edge_attr, 
                distmat_top, distmat_3d,
                batch,
                return_dense=True):

        # process node
        x=self.residue_enc(x)
        sasa=self.sasa_enc(torch.nan_to_num(sasa_residue_wise) )
        x = torch.cat( (x,sasa), 1)
        pe= self.pe_lin(residue_ids,chain_ids) 
        rots=self.rotamers_enc(rotamers)
        covalent_bonds_attr=self.res_res_covalent_bonds_enc(covalent_edge_attr)
        Bakbone_dihs=self.Bakbone_dihs_enc(res_res_BB_dihs)
        #print(x.shape, sasa.shape, pe.shape, rots.shape )
        if self.emb_add:
            x+=  rots + pe
            covalent_bonds_attr+= Bakbone_dihs
        else:            
            x = torch.cat((x,rots,pe), 1)
            covalent_bonds_attr= torch.cat(covalent_bonds_attr, Bakbone_dihs)
        Non_covalent_edge_attr =self.res_res_Non_covalent_bonds_enc(noncovalent_edge_attr)
        total_edge_attr = torch.cat((covalent_bonds_attr,Non_covalent_edge_attr ), 0)
        #total_edge_index = torch.cat((edge_index,noncovalent_edge_index ), 0)
        # get the sph
        sph = process_hop(distmat_3d, self.gamma, self.hop, self.slope)
        #print('x=',x.size(),'edge_index',edge_index.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        #print(covalent_bonds_attr.shape, Non_covalent_edge_attr.shape, edge_index.shape, noncovalent_edge_index.shape )
        # Conv & MPNN
        if self.mpnn == 'GIN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch)
        elif self.mpnn == 'GCN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=total_edge_attr, pe=pe)
        elif self.mpnn == 'GINE':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=total_edge_attr)
        #print("x before pooling=", x.size())
        if return_dense:
            h, nomask = to_dense_batch(x, batch)
            return h, nomask
            #h.masked_fill(~nomask.unsqueeze(-1), float('-inf'))
            #from torch_geometric.utils import mask_select, mask_to_index, select

        else:
            return x

class mol_residues_light_level_encoder(torch.nn.Module):
    def __init__(self, hidden_dim=100, sasa_hidden_dim=10,
                 emb_add=True, ## add the emb for pe and rotamers to (residues,sasa) and res_res_BondEncoder+ Bakbone_dihs_enc
                 rotamers_hidden_dim=20,   pe_dim=20 ,# ignore with emb_add
                 mpnn='GINE',
                 gamma=0.5, slope=0.0, hops_list=None,
                 nhead=4,n_hop=5,num_layers=10, 
                 dropout=0.1, drop_prob=0.0, attn_dropout=0.1
                ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.sasa_enc=Linear(4, sasa_hidden_dim)
        self.emb_add=emb_add
        if emb_add:
            self.pe_lin=Residues_LearnedPositionalEmbedding(emb_dim=hidden_dim)
            #self.rotamers_enc= dihedral_angles_Embedding_Encoder( emb_dim=hidden_dim,max_nr_dihs=4)
            self.residue_enc=ResidueEncoder( emb_dim=hidden_dim-sasa_hidden_dim, add_sasas=False)
            ### bonded interaction
            self.res_res_covalent_bonds_enc=res_res_BondEncoder(emb_dim=hidden_dim)
            #self.Bakbone_dihs_enc=dihedral_angles_Embedding_Encoder(emb_dim=hidden_dim,max_nr_dihs=2 )
        else:
            self.pe_lin=Residues_LearnedPositionalEmbedding(emb_dim=pe_dim)
            #self.rotamers_enc= dihedral_angles_Embedding_Encoder( emb_dim=rotamers_hidden_dim)
            self.residue_enc=ResidueEncoder( emb_dim=hidden_dim- pe_dim-rotamers_hidden_dim -sasa_hidden_dim, add_sasas=False)
            ### bonded interaction
            res_res_Bond_dim=hidden_dim//2
            self.res_res_covalent_bonds_enc=res_res_BondEncoder(emb_dim=res_res_Bond_dim)
            #self.Bakbone_dihs_enc=dihedral_angles_Embedding_Encoder(emb_dim=hidden_dim - res_res_Bond_dim)
        self.res_res_Non_covalent_bonds_enc=res_res_Non_covalent_BondEncoder(emb_dim=hidden_dim)
        self.gamma = gamma
        self.slope = slope
        self.mpnn = mpnn
        if hops_list:
            assert len(hops_list)== nhead
            self.hop = Parameter(torch.tensor(hops_list).reshape([nhead,1,1]))
        else:
            self.hop = Parameter(torch.full((nhead, 1, 1), float(n_hop)))
        self.convs = ModuleList()
        if mpnn == 'GIN':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GINE':
            for _ in range(num_layers):
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim),
                )
                conv = GPSConv(hidden_dim, GINEConv(nn), heads=nhead, dropout=dropout,
                               attn_dropout=attn_dropout, drop_prob=drop_prob)
                self.convs.append(conv)
        elif mpnn == 'GCN':
            #Lap_pe = False
            for _ in range(num_layers):
                conv = GPSConv(hidden_dim, GatedGCNLayer(channels, channels, 0, True,
                                                            equivstable_pe=Lap_pe),
                               heads=nhead, dropout=dropout, attn_dropout=attn_dropout,
                               drop_prob=drop_prob)
                self.convs.append(conv)

    def forward(self, x, sasa_residue_wise , rotamers, residue_ids,chain_ids, edge_index,
                covalent_edge_index, covalent_edge_attr,res_res_BB_dihs,
                noncovalent_edge_index, noncovalent_edge_attr, 
                distmat_top, distmat_3d,
                batch,
                return_dense=True):

        # process node
        x=self.residue_enc(x)
        sasa=self.sasa_enc(torch.nan_to_num(sasa_residue_wise) )
        x = torch.cat((x, sasa), 1)
        pe= self.pe_lin(residue_ids,chain_ids) 
        #rots=self.rotamers_enc(rotamers)
        covalent_bonds_attr=self.res_res_covalent_bonds_enc(covalent_edge_attr)
        #Bakbone_dihs=self.Bakbone_dihs_enc(res_res_BB_dihs)
        #print(x.shape, sasa.shape, pe.shape, rots.shape )
        if self.emb_add:
            #x+=  rots + pe
            #covalent_bonds_attr+= Bakbone_dihs
            x+=pe
        else:            
            #x = torch.cat((x,rots,pe), 1)
            x = torch.cat((x,pe), 1)
            covalent_bonds_attr= torch.cat(covalent_bonds_attr, Bakbone_dihs)
        Non_covalent_edge_attr =self.res_res_Non_covalent_bonds_enc(noncovalent_edge_attr)

        total_edge_attr = torch.cat((covalent_bonds_attr,Non_covalent_edge_attr ), 0)
        #total_edge_index = torch.cat((edge_index,noncovalent_edge_index ), 0)
        # get the sph
        sph = process_hop(distmat_3d, self.gamma, self.hop, self.slope)

        #print('x=',x.size(),'edge_index',edge_index.size(),'edge_attr', edge_attr.size(),'sph',sph.size())
        #print(covalent_bonds_attr.shape, Non_covalent_edge_attr.shape, edge_index.shape, noncovalent_edge_index.shape )
        # Conv & MPNN
        if self.mpnn == 'GIN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch)
        elif self.mpnn == 'GCN':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=total_edge_attr, pe=pe)
        elif self.mpnn == 'GINE':
            for conv in self.convs:
                x = conv(x, edge_index, sph, batch, edge_attr=total_edge_attr)
        #print("x before pooling=", x.size())
        if return_dense:
            h, nomask = to_dense_batch(x, batch)
            return h, nomask
            #h.masked_fill(~nomask.unsqueeze(-1), float('-inf'))
            #from torch_geometric.utils import mask_select, mask_to_index, select

        else:
            return x





###################################################################################
####################
def process_hop(sph, gamma, hop, slope=0.1):
    leakyReLU = LeakyReLU(negative_slope=slope)
    sph = sph.unsqueeze(1)
    sph = sph - hop
    sph = leakyReLU(sph)
    sp = torch.pow(gamma, sph)
    return sp
#############################################################################################