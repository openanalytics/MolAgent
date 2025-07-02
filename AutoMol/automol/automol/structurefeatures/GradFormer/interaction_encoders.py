import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data,Batch
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
#from add_path import algos
import algos


class InteractionMHAType(torch.nn.Module):
    def __init__(self,emb_dim,param_indices=[0], nhead=6,n_layers=10,max_interact=50):
        super(InteractionMHAType, self).__init__()
        self.n_features=my_special_round(2*emb_dim+len(param_indices),nhead)
        self.padding_index= self.n_features-(2*emb_dim+len(param_indices))
        self.param_indices=param_indices
        self.max_interact=max_interact
        self.linear1=torch.nn.Linear(self.n_features,1)
        self.linear2=torch.nn.Linear(self.max_interact,1)
        encoder_layer= torch.nn.TransformerEncoderLayer(d_model=self.n_features,nhead=nhead,batch_first=True)
        self.transformer_encoder=torch.nn.TransformerEncoder(encoder_layer,num_layers=n_layers)
        
    def forward(self, params,pro,lig):
        out=torch.nn.functional.pad(torch.cat([params[:,:,self.param_indices],pro,lig ], dim=2) ,(0,self.padding_index),value=0.)
        interac_pad_val=self.max_interact-params.shape[1]
        return self.linear2(torch.nn.functional.pad(self.linear1(self.transformer_encoder(out)).squeeze(dim=2),(0,interac_pad_val),value=0. ))
    
class InteractionMatMulType(torch.nn.Module):
    def __init__(self,emb_dim,param_indices=[0],max_interact=50):
        super(InteractionMatMulType, self).__init__()
        self.param_indices=param_indices
        self.max_interact=max_interact
        self.activation=torch.nn.Sigmoid()
        self.linear1=torch.nn.Linear(self.max_interact,1)
        self.linear2=torch.nn.Linear(self.max_interact,1)
        self.linear3=torch.nn.Linear(len(self.param_indices),1)
        
    def forward(self, params,pro,lig):
        #print(params.shape,pro.shape,lig.shape)
        interac_pad_val=self.max_interact-params.shape[1]
        mm=torch.nn.functional.pad(self.activation(torch.bmm(pro, lig.transpose(1,2))),(0,interac_pad_val,0,interac_pad_val),value=0. )
        param_act=torch.nn.functional.pad(self.activation(self.linear3(params[:,:,self.param_indices])).squeeze(dim=2),(0,interac_pad_val),value=0. )
        return self.linear1(self.linear2(mm).squeeze(dim=2)*param_act)
    
class InteractionMatMulExpdistanceType(torch.nn.Module):
    def __init__(self,emb_dim,param_indices=[0],max_interact=50,hidden_dropout=0.1):
        super(InteractionMatMulExpdistanceType, self).__init__()
        self.param_indices=param_indices
        self.max_interact=max_interact
        self.activation=torch.nn.Sigmoid()
        self.linear1=torch.nn.Linear(self.max_interact,1)
        self.linear2=torch.nn.Linear(self.max_interact,1)
        self.linear3=torch.nn.Linear(len(self.param_indices),1)
        self.param_norm=torch.nn.LayerNorm(len(self.param_indices))
        self.mm_norm=torch.nn.LayerNorm(max_interact)
        self.decay=torch.nn.Parameter(torch.ones(1))
        self.dropout = torch.nn.Dropout(hidden_dropout)
        
    def forward(self, params,pro,lig):
        #print(params.shape,pro.shape,lig.shape)
        interac_pad_val=self.max_interact-params.shape[1]
        mm=torch.nn.functional.pad(torch.bmm(pro, lig.transpose(1,2)),(0,interac_pad_val,0,interac_pad_val),value=0. )
        param_act=torch.nn.functional.pad(self.activation(torch.exp(-self.decay*params[:,:,0])*(self.linear3(self.param_norm(params[:,:,self.param_indices])).squeeze(dim=2))),(0,interac_pad_val),value=0. )
        return self.linear1(self.dropout(self.activation(self.linear2(self.mm_norm(mm)))).squeeze(dim=2)*param_act)
    
class InteractionMatMulParamEmbType(torch.nn.Module):
    def __init__(self,emb_dim,param_emb_dim,max_interact=50):
        super(InteractionMatMulParamEmbType, self).__init__()
        self.param_emb_dim=param_emb_dim
        self.max_interact=max_interact
        self.activation=torch.nn.Sigmoid()
        self.linear1=torch.nn.Linear(self.max_interact,1)
        self.linear2=torch.nn.Linear(self.max_interact,1)
        self.linear3=torch.nn.Linear(param_emb_dim,1)
        
    def forward(self, params,pro,lig):
        #print(params.shape,pro.shape,lig.shape)
        interac_pad_val=self.max_interact-pro.shape[1]
        mm=torch.nn.functional.pad(self.activation(torch.bmm(pro, lig.transpose(1,2))),(0,interac_pad_val,0,interac_pad_val),value=0. )
        param_act=torch.nn.functional.pad(self.activation(self.linear3(params)).squeeze(dim=2),(0,interac_pad_val),value=0. )
        #print(mm.shape,param_act.shape,pro.shape,params.shape,self.linear3(params).shape )
        return self.linear1(self.linear2(mm).squeeze(dim=2)*param_act)
    
    

class ParameterEmbedding(torch.nn.Module):
    def __init__(self, emb_dim=10,val_range=[0.,1.], bin_interval=0.2):
        super(ParameterEmbedding, self).__init__()
        self.hidden_dim=emb_dim
        self.bins=torch.range(val_range[0],val_range[1],bin_interval)
        self.padding= len(self.bins)
        embd_size=len(self.bins)+1 
        #print('self.padding',self.padding,embd_size)
        self.param_emb= torch.nn.Embedding(embd_size, emb_dim, padding_idx=self.padding)
        torch.nn.init.xavier_uniform_(self.param_emb.weight.data)
    def forward(self, param):
        device = param.device
        mask=torch.isnan(param).to(device=device)
        position_ids=torch.bucketize(param,self.bins.to(device=device))
        position_ids[mask]=self.padding
        x_embedding =self.param_emb(position_ids)
        return x_embedding


class FullParameterEmbedding(torch.nn.Module):
    def __init__(self, emb_dim=10,
                interactions_parameters_names_ranges={'distance':{'range':[0,7], 'interval':0.2},
                                      'DHA_angle': {'range':[120,180], 'interval':5.0},
                                      "AXD_angle": {'range':[70,180], 'interval':5.0},#(130, 180)
                                      "XAR_angle": {'range':[70,150], 'interval':5.0}, 
                                      'plane_angle': {'range':[0,95], 'interval':5.0}, #plane_angle=(0, 35) , (50, 90),
                                      'normal_to_centroid_angle' : {'range':[0,40], 'interval':5.0},#(0, 33)
                                      'intersect_radius':None , # ignore .. fixed par        intersect_radius=1.5,
                                       'intersect_distance':{'range':[0,2], 'interval':0.2},# 1.5
                                      'angle_Cation_Pi': {'range':[70,150], 'interval':5.0}#(0, 30)
                                     }):
        super(FullParameterEmbedding, self).__init__()
        self.emb_dim=emb_dim
        self.used_indices=[ idx for idx,(key,item) in enumerate(interactions_parameters_names_ranges.items()) if item is not None]
        self.embeddings_list=torch.nn.ModuleList([ParameterEmbedding(emb_dim,item['range'],item['interval']) for key,item in interactions_parameters_names_ranges.items() if item is not None])
    
    def forward(self,params):
        return torch.cat([self.embeddings_list[idx].forward(params[:,:,index]) for idx,index in enumerate(self.used_indices)],2) 
    
class FullParameterDistanceEmbedding(torch.nn.Module):
    def __init__(self, emb_dim=10,
                interactions_parameters_names_ranges={'distance':{'range':[0,7], 'interval':0.2},
                                      'DHA_angle': {'range':[120,180], 'interval':5.0},
                                      "AXD_angle": {'range':[70,180], 'interval':5.0},#(130, 180)
                                      "XAR_angle": {'range':[70,150], 'interval':5.0}, 
                                      'plane_angle': {'range':[0,95], 'interval':5.0}, #plane_angle=(0, 35) , (50, 90),
                                      'normal_to_centroid_angle' : {'range':[0,40], 'interval':5.0},#(0, 33)
                                      'intersect_radius':None , # ignore .. fixed par        intersect_radius=1.5,
                                       'intersect_distance':{'range':[0,2], 'interval':0.2},# 1.5
                                      'angle_Cation_Pi': {'range':[70,150], 'interval':5.0}#(0, 30)
                                     }):
        super(FullParameterDistanceEmbedding, self).__init__()
        self.emb_dim=emb_dim
        self.used_indices=[ idx for idx,(key,item) in enumerate(interactions_parameters_names_ranges.items()) if item is not None]
        self.embeddings_list=torch.nn.ModuleList([ParameterEmbedding(emb_dim,item['range'],item['interval']) for key,item in interactions_parameters_names_ranges.items() if item is not None])
        self.decay=torch.nn.Parameter(torch.ones(1))
        self.lnorm=torch.nn.LayerNorm(1)
        self.activation = torch.nn.Sigmoid()
        
    
    def forward(self,params):
        dist_mask=params[:,:,0]<0
        params[dist_mask]=1e9
        exp_min_dist=torch.exp(-self.decay*params[:,:,0])        
        return self.lnorm(exp_min_dist.unsqueeze(dim=2))*torch.cat([self.embeddings_list[idx].forward(params[:,:,index]) for idx,index in enumerate(self.used_indices)],2) 
    
    
    

class InteractionPharmacoPhoreEncoder(torch.nn.Module):
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder, nb_interac_types=9,
                 param_indices_l=[[0],[0],[0],[0,8],[0,1],[0,4,5,7],[0,4,5],[0],[0,2,3]] ,
                 nhead=6,n_layers=10, expdecay=True):
        super(InteractionPharmacoPhoreEncoder, self).__init__()
        self.hidden_dim = emb_dim
        self.protein_enc = protein_enc
        self.ligand_enc = ligand_enc
        self.param_encoder=param_encoder
        self.nb_interac_types = nb_interac_types
        self.param_indices_l=param_indices_l
        #self.interaction_modules=torch.nn.ModuleList([InteractionMHAType(self.hidden_dim,ind,nhead,n_layers) for ind in self.param_indices_l])
        #
        if self.param_encoder is not None:
            self.interaction_modules=torch.nn.ModuleList([InteractionMatMulParamEmbType(self.hidden_dim,param_encoder.emb_dim*len(param_encoder.used_indices)) for ind in self.param_indices_l])
        elif expdecay:
            self.interaction_modules=torch.nn.ModuleList([InteractionMatMulExpdistanceType(self.hidden_dim,ind) for ind in self.param_indices_l])
        else:
            self.interaction_modules=torch.nn.ModuleList([InteractionMatMulType(self.hidden_dim,ind) for ind in self.param_indices_l])
        self.layer_norm = torch.nn.LayerNorm(4)
        
        
        
    def forward_ligand_model(self,batch,key):
        node_num = batch[key].topological_dist_mat_values.shape[-1]
        batch[key].sph = batch[key]['topological_dist_mat_values'].reshape(-1, node_num, node_num)
        h_ligand, nomask = self.ligand_enc(x=batch[key].x,
                                        sasa= batch[key].sasa_atom_wise,
                                        pe=batch[key].pe,
                                        edge_index=batch[key].edge_index,
                                        edge_attr=batch[key].edge_attr,
                                        batch=batch[key].batch, 
                                        sph=batch[key].sph)
        return h_ligand, nomask
        
    def forward_interactions(self,batch,index_key,h_prot,h_ligand):
        
        interaction_out,sasa_out=[],[]
        for  key in ['ligand_delta_sasa_Polar','ligand_delta_sasa_Apolar','protein_delta_sasa_Polar','protein_delta_sasa_Apolar']:
            sasa_out.append(torch.sum(batch['interactions'+index_key][key],dim=1).unsqueeze(dim=1))
            #sasa_out.append(torch.mean(batch['interactions'][key],dim=1,keepdim=True).to(self.device))  
            
        norm_sasa=self.layer_norm(torch.hstack(sasa_out))
        protein_dict={ pi.item():bi for bi,pi in enumerate(batch['protein_indices'])}
        
        param_enc=None
        if self.param_encoder is not None:
            param_enc=self.param_encoder.forward(batch['interactions'+index_key]['interactions_parameters'])
        
        for c in range(self.nb_interac_types):
            #check one_hot encoding for interaction c
            interaction_mask=(batch['interactions'+index_key]['interactions_One_hot_encoding'][:,:,c]==1)
            #interaction pairs per ligand
            nodes_per_ligand=[0]+[sum([torch.sum(fi) for fi in interaction_mask[:i+1]]) for i in range(len(interaction_mask))]
            
            #parameters
            if param_enc is not None:
                masked_params=param_enc[interaction_mask]
                batch_param=[masked_params[start:end,:] for start,end in zip(nodes_per_ligand[:-1],nodes_per_ligand[1:])]
            else:
                masked_params=batch['interactions'+index_key]['interactions_parameters'][interaction_mask]
                batch_param=[masked_params[start:end,:] for start,end in zip(nodes_per_ligand[:-1],nodes_per_ligand[1:])]
            
            #lig pairs
            masked_pairs_lig=batch['interactions'+index_key]['interactions_pairs_lig'][interaction_mask]
            batch_masked_pairs_lig=[masked_pairs_lig[start:end] for start,end in zip(nodes_per_ligand[:-1],nodes_per_ligand[1:])]
            
            #lig pairs
            masked_pairs_pro=batch['interactions'+index_key]['interactions_pairs_pro'][interaction_mask]
            batch_masked_pairs_pro=[masked_pairs_pro[start:end] for start,end in zip(nodes_per_ligand[:-1],nodes_per_ligand[1:])]
            
            #corresponding protein embeddings
            pro_emb=[ h_prot[protein_dict[val.item()],pro_p,:] for val,pro_p in zip(batch['indices'+index_key][:,0],batch_masked_pairs_pro)]
            
            #corresponding ligand embeddings
            ligand_emb=[ h_ligand[b,lig_p,:] for b,lig_p in enumerate(batch_masked_pairs_lig)]

            interaction_out.append(self.interaction_modules[c].forward(pad_sequence(batch_param, batch_first=True, padding_value=0.0),
                                                           pad_sequence(pro_emb, batch_first=True, padding_value=0.0),
                                                           pad_sequence(ligand_emb, batch_first=True, padding_value=0.0)))
        return torch.hstack([norm_sasa]+interaction_out)  

class InteractionAbsolutePharmacoPhoreEncoder(InteractionPharmacoPhoreEncoder): 
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder, nb_interac_types=9,
                 param_indices_l=[[0],[0],[0],[0,8],[0,1],[0,4,5,7],[0,4,5],[0],[0,2,3]] ,
                 nhead=6,n_layers=10, expdecay=True):
        super(InteractionAbsolutePharmacoPhoreEncoder, self).__init__(emb_dim,protein_enc,ligand_enc,param_encoder, nb_interac_types,
                 param_indices_l,
                 nhead,n_layers,expdecay)
        
    def forward(self, batch):
        
        #protein
        node_num = batch['protein'].centriods_dists.shape[-1]
        batch['protein'].centriods_dists = batch['protein'].centriods_dists.reshape(-1, node_num, node_num)
        h_prot,nomask=self.protein_enc( batch['protein'].x,
                                       batch['protein'].sasa_residue_wise,
                                       batch['protein'].rotamers,
                                       batch['protein'].residue_ids,
                                       batch['protein'].chain_ids,
                                       batch['protein'].edge_index,
                                       batch['protein'].covalent_edge_index,
                                       batch['protein'].covalent_edge_attr,
                                       batch['protein'].res_res_BB_dihs,
                                       batch['protein'].noncovalent_edge_index,
                                       batch['protein'].noncovalent_edge_attr,
                                       distmat_top=None,
                                       distmat_3d= batch['protein'].centriods_dists,
                                       batch=batch['protein'].batch,
                                       return_dense=True)
        
        h_ligand_one, nomask = self.forward_ligand_model(batch,'ligands')
        
        interactions_one = self.forward_interactions(batch,'',h_prot,h_ligand_one)
        
        return interactions_one
    
class InteractionRelativePharmacoPhoreEncoder(InteractionPharmacoPhoreEncoder): 
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder, nb_interac_types=9,
                 param_indices_l=[[0],[0],[0],[0,8],[0,1],[0,4,5,7],[0,4,5],[0],[0,2,3]] ,
                 nhead=6,n_layers=10, expdecay=True):
        super(InteractionRelativePharmacoPhoreEncoder, self).__init__(emb_dim,protein_enc,ligand_enc,param_encoder, nb_interac_types,
                 param_indices_l,
                 nhead,n_layers,expdecay)
        
    def forward(self, batch):
        
        #protein
        node_num = batch['protein'].centriods_dists.shape[-1]
        batch['protein'].centriods_dists = batch['protein'].centriods_dists.reshape(-1, node_num, node_num)
        h_prot,nomask=self.protein_enc( batch['protein'].x,
                                       batch['protein'].sasa_residue_wise,
                                       batch['protein'].rotamers,
                                       batch['protein'].residue_ids,
                                       batch['protein'].chain_ids,
                                       batch['protein'].edge_index,
                                       batch['protein'].covalent_edge_index,
                                       batch['protein'].covalent_edge_attr,
                                       batch['protein'].res_res_BB_dihs,
                                       batch['protein'].noncovalent_edge_index,
                                       batch['protein'].noncovalent_edge_attr,
                                       distmat_top=None,
                                       distmat_3d= batch['protein'].centriods_dists,
                                       batch=batch['protein'].batch,
                                       return_dense=True)
        
        h_ligand_one, nomask = self.forward_ligand_model(batch,'ligands_one')
        h_ligand_two, nomask = self.forward_ligand_model(batch,'ligands_two')
        
        interactions_one = self.forward_interactions(batch,'_one',h_prot,h_ligand_one)
        interactions_two = self.forward_interactions(batch,'_two',h_prot,h_ligand_two)
        
        return torch.hstack([interactions_one,interactions_two,interactions_one-interactions_two])
    
    

class InteractionGraphEncoder(torch.nn.Module):
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder,interaction_encoder,pre_transform,inc_padding_val=-510):
        super(InteractionGraphEncoder, self).__init__()
        self.hidden_dim = emb_dim
        self.protein_enc = protein_enc
        self.ligand_enc = ligand_enc
        self.param_encoder = param_encoder
        self.interaction_encoder = interaction_encoder
        self.pre_transform=pre_transform
        self.inc_padding_val=inc_padding_val
        #self.layer_norm = torch.nn.LayerNorm(4)
        
    def forward_ligand_model(self,batch,key):
        node_num = batch[key].topological_dist_mat_values.shape[-1]
        batch[key].sph = batch[key]['topological_dist_mat_values'].reshape(-1, node_num, node_num)
        h_ligand, nomask = self.ligand_enc(x=batch[key].x,
                                        sasa= batch[key].sasa_atom_wise,
                                        pe=batch[key].pe,
                                        edge_index=batch[key].edge_index,
                                        edge_attr=batch[key].edge_attr,
                                        batch=batch[key].batch, 
                                        sph=batch[key].sph)
        return h_ligand, nomask
        
    def forward_interactionGraph(self,batch,index_key,h_prot,h_ligand):
        param_enc=self.param_encoder.forward(batch['interactions'+index_key]['interactions_parameters'])
        #interaction_out,sasa_out=[],[]  
        #TODO: use individual sasa
        #for  key in ['ligand_delta_sasa_Polar','ligand_delta_sasa_Apolar','protein_delta_sasa_Polar','protein_delta_sasa_Apolar']:
        #    sasa_out.append(torch.sum(batch['interactions'+index_key][key],dim=1).to(self.device).unsqueeze(dim=1))
        #norm_sasa=self.layer_norm(torch.hstack(sasa_out))
        
        device = h_prot.device
        
        data_list = []
        sph_l=[]
        for b in range(h_ligand.shape[0]):
            data = Data()

            protein_idx=(batch['protein_indices']==batch['indices'+index_key][b,0]).nonzero().squeeze(1)[0]  
            indices_lig=batch['interactions'+index_key]['interactions_pairs_lig'][b][batch['interactions'+index_key]['interactions_pairs_lig'][b]!=self.inc_padding_val]
            indices_prot=batch['interactions'+index_key]['interactions_pairs_pro'][b][batch['interactions'+index_key]['interactions_pairs_pro'][b]!=self.inc_padding_val]
            prot_nodes=h_prot[protein_idx,indices_prot,:]
            lig_nodes=h_ligand[b,indices_lig,:]

            num_interactions=len(indices_lig)
            data.x=torch.vstack([prot_nodes,lig_nodes])
            data.__num_nodes__ =data.x.shape[0]
            protein_sasa=torch.hstack( [batch['interactions'+index_key]['protein_delta_sasa_Polar'][b,indices_prot],batch['interactions'+index_key]['protein_delta_sasa_Apolar'][b,indices_prot] ])
            ligand_sasa=torch.hstack( [batch['interactions'+index_key]['ligand_delta_sasa_Polar'][b,indices_lig],batch['interactions'+index_key]['ligand_delta_sasa_Apolar'][b,indices_lig] ])

            data.sasa=torch.vstack([protein_sasa,ligand_sasa]).transpose(0,1)

            #nodes corresponds to node features, protein first half, ligand second half
            edges_list=[ (i,i+num_interactions) for i in range(num_interactions)]+[ (i+num_interactions,i) for i in range(num_interactions)]
            edge_index=np.array(edges_list, dtype = np.int64).T
            data.edge_index = torch.from_numpy(edge_index).to(dtype=torch.int64).to(device=device)

            edge_attr=param_enc[b,:num_interactions,:]
            data.edge_attr =torch.vstack([edge_attr,edge_attr])
            N = data.x.shape[0]
            sp = torch.zeros([N, N]).to(device=device)
            sp[data.edge_index[0, :], data.edge_index[1, :]] = 1.0
            
            sp[:num_interactions,:num_interactions]=batch['protein'].centriods_dists[protein_idx][indices_prot,indices_prot]
            sp[num_interactions:,num_interactions:]=batch['ligands'+index_key].sph[b][indices_lig,indices_lig]
            #sp, _ = torch.tensor(algos.floyd_warshall(adj.numpy()))
            #data['sph']=sp
            sph_l.append(sp)
            data=self.pre_transform(data)
            data_list.append(data)
        
        max_num_nodes = max([data.num_nodes for data in data_list])
        sph_list=[]
        for data,sp in zip(data_list,sph_l):
            num_nodes = data.num_nodes
            pad_size = max_num_nodes - num_nodes
            padded_sph= torch.nn.functional.pad(sp, (0, pad_size, 0, pad_size), value=510)
            sph_list.append(padded_sph)
        
        interactionbatch=Batch.from_data_list(data_list)
        
        h_inter, nomask_intera = self.interaction_encoder(x=interactionbatch.x,
                                        sasa=interactionbatch.sasa,
                                        pe=interactionbatch.pe,
                                        edge_index=interactionbatch.edge_index,
                                        edge_attr=interactionbatch.edge_attr,
                                        batch=interactionbatch.batch, 
                                        sph=torch.stack(sph_list))
        
        return h_inter, nomask_intera
      

class InteractionAbsoluteGraphEncoder(InteractionGraphEncoder):
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder,interaction_encoder,pre_transform,inc_padding_val=-510):
        super(InteractionAbsoluteGraphEncoder, self).__init__(emb_dim,protein_enc,ligand_enc,param_encoder,interaction_encoder,pre_transform,inc_padding_val)
        
    def forward(self, batch):
        node_num = batch['protein'].centriods_dists.shape[-1]
        batch['protein'].centriods_dists = batch['protein'].centriods_dists.reshape(-1, node_num, node_num)
        
        h_prot,nomask=self.protein_enc( batch['protein'].x,
                                       batch['protein'].sasa_residue_wise,
                                       batch['protein'].rotamers,
                                       batch['protein'].residue_ids,
                                       batch['protein'].chain_ids,
                                       batch['protein'].edge_index,
                                       batch['protein'].covalent_edge_index,
                                       batch['protein'].covalent_edge_attr,
                                       batch['protein'].res_res_BB_dihs,
                                       batch['protein'].noncovalent_edge_index,
                                       batch['protein'].noncovalent_edge_attr,
                                       distmat_top=None,
                                       distmat_3d= batch['protein'].centriods_dists,
                                       batch=batch['protein'].batch,
                                       return_dense=True)
        
        h_ligand_one, nomask = self.forward_ligand_model(batch,'ligands')
        
        interactionGraph_one, nomask_one = self.forward_interactionGraph(batch,'',h_prot,h_ligand_one)
        
        return interactionGraph_one, nomask_one
        
class InteractionRelativeGraphEncoder(InteractionGraphEncoder):
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder,interaction_encoder,pre_transform,inc_padding_val=-510):
        super(InteractionRelativeGraphEncoder, self).__init__(emb_dim,protein_enc,ligand_enc,param_encoder,interaction_encoder,pre_transform,inc_padding_val)
        
    def forward(self, batch):
        node_num = batch['protein'].centriods_dists.shape[-1]
        batch['protein'].centriods_dists = batch['protein'].centriods_dists.reshape(-1, node_num, node_num)
        
        h_prot,nomask=self.protein_enc( batch['protein'].x,
                                       batch['protein'].sasa_residue_wise,
                                       batch['protein'].rotamers,
                                       batch['protein'].residue_ids,
                                       batch['protein'].chain_ids,
                                       batch['protein'].edge_index,
                                       batch['protein'].covalent_edge_index,
                                       batch['protein'].covalent_edge_attr,
                                       batch['protein'].res_res_BB_dihs,
                                       batch['protein'].noncovalent_edge_index,
                                       batch['protein'].noncovalent_edge_attr,
                                       distmat_top=None,
                                       distmat_3d= batch['protein'].centriods_dists,
                                       batch=batch['protein'].batch,
                                       return_dense=True)
        
        h_ligand_one, nomask = self.forward_ligand_model(batch,'ligands_one')
        h_ligand_two, nomask = self.forward_ligand_model(batch,'ligands_two')
        
        interactionGraph_one, nomask_one = self.forward_interactionGraph(batch,'_one',h_prot,h_ligand_one)
        interactionGraph_two, nomask_two = self.forward_interactionGraph(batch,'_two',h_prot,h_ligand_two)
        
        return interactionGraph_one, nomask_one, interactionGraph_two, nomask_two
    
#############################################################################################    
class InteractionLigandEncoder(torch.nn.Module):
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder,pred_hidden_dropout=0.2,inc_padding_val=-510):
        super(InteractionLigandEncoder, self).__init__()
        self.hidden_dim = emb_dim
        self.protein_enc = protein_enc
        self.ligand_enc = ligand_enc
        self.param_encoder = param_encoder
        self.linear1=torch.nn.Linear(param_encoder.emb_dim*len(param_encoder.used_indices),1)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(pred_hidden_dropout)
        self.inc_padding_val=inc_padding_val
        #self.layer_norm = torch.nn.LayerNorm(4)
        
        
    def forward_ligand_model(self,batch,index_key,h_prot):
        param_enc=self.param_encoder.forward(batch['interactions'+index_key]['interactions_parameters'])
        #interaction_out,sasa_out=[],[]  
        #TODO: use individual sasa
        #for  key in ['ligand_delta_sasa_Polar','ligand_delta_sasa_Apolar','protein_delta_sasa_Polar','protein_delta_sasa_Apolar']:
        #    sasa_out.append(torch.sum(batch['interactions'+index_key][key],dim=1).to(self.device).unsqueeze(dim=1))
        #norm_sasa=self.layer_norm(torch.hstack(sasa_out))
        
        device = h_prot.device
        
        data_list = []
        sph_l=[]
        protein_inter=torch.zeros(batch['ligands'+index_key].x.shape[0],h_prot.shape[2]).to(device=device)
        split=0
        for b in range(batch['ligands'+index_key].ligand_affinity.shape[0]):

            protein_idx=(batch['protein_indices']==batch['indices'+index_key][b,0]).nonzero().squeeze(1)[0]  
            indices_lig=batch['interactions'+index_key]['interactions_pairs_lig'][b][batch['interactions'+index_key]['interactions_pairs_lig'][b]!=self.inc_padding_val]
            indices_prot=batch['interactions'+index_key]['interactions_pairs_pro'][b][batch['interactions'+index_key]['interactions_pairs_pro'][b]!=self.inc_padding_val]
            num_interactions=len(indices_lig)
            for i_l, i_p, i in zip(indices_lig,indices_prot,range(num_interactions)):
                protein_inter[split+i_l,:]+=self.activation(self.linear1(self.dropout(param_enc[b,i,:])))*h_prot[protein_idx,i_p,:]

            split+=sum(batch['ligands'+index_key].batch==b)
        
        node_num = batch['ligands'+index_key].topological_dist_mat_values.shape[-1]
        batch['ligands'+index_key].sph = batch['ligands'+index_key]['topological_dist_mat_values'].reshape(-1, node_num, node_num)
        h_inter, nomask_intera = self.ligand_enc(x=batch['ligands'+index_key].x,
                                         res_enc=protein_inter,
                                        sasa= batch['ligands'+index_key].sasa_atom_wise,
                                        pe=batch['ligands'+index_key].pe,
                                        edge_index=batch['ligands'+index_key].edge_index,
                                        edge_attr=batch['ligands'+index_key].edge_attr,
                                        batch=batch['ligands'+index_key].batch, 
                                        sph=batch['ligands'+index_key].sph)
        
        return h_inter, nomask_intera
      

class InteractionAbsoluteLigandEncoder(InteractionLigandEncoder):
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder,inc_padding_val=-510):
        super(InteractionAbsoluteLigandEncoder, self).__init__(emb_dim,protein_enc,ligand_enc,param_encoder,inc_padding_val=inc_padding_val)
        
    def forward(self, batch):
        node_num = batch['protein'].centriods_dists.shape[-1]
        batch['protein'].centriods_dists = batch['protein'].centriods_dists.reshape(-1, node_num, node_num)
        
        h_prot,nomask=self.protein_enc( batch['protein'].x,
                                       batch['protein'].sasa_residue_wise,
                                       batch['protein'].rotamers,
                                       batch['protein'].residue_ids,
                                       batch['protein'].chain_ids,
                                       batch['protein'].edge_index,
                                       batch['protein'].covalent_edge_index,
                                       batch['protein'].covalent_edge_attr,
                                       batch['protein'].res_res_BB_dihs,
                                       batch['protein'].noncovalent_edge_index,
                                       batch['protein'].noncovalent_edge_attr,
                                       distmat_top=None,
                                       distmat_3d= batch['protein'].centriods_dists,
                                       batch=batch['protein'].batch,
                                       return_dense=True)
        
        h_ligand_one, nomask = self.forward_ligand_model(batch,'',h_prot)
                
        return h_ligand_one, nomask
        
class InteractionRelativeLigandEncoder(InteractionLigandEncoder):
    def __init__(self, emb_dim,protein_enc,ligand_enc,param_encoder,inc_padding_val=-510):
        super(InteractionRelativeLigandEncoder, self).__init__(emb_dim,protein_enc,ligand_enc,param_encoder,inc_padding_val=inc_padding_val)
        
    def forward(self, batch):
        node_num = batch['protein'].centriods_dists.shape[-1]
        batch['protein'].centriods_dists = batch['protein'].centriods_dists.reshape(-1, node_num, node_num)
        
        h_prot,nomask=self.protein_enc( batch['protein'].x,
                                       batch['protein'].sasa_residue_wise,
                                       batch['protein'].rotamers,
                                       batch['protein'].residue_ids,
                                       batch['protein'].chain_ids,
                                       batch['protein'].edge_index,
                                       batch['protein'].covalent_edge_index,
                                       batch['protein'].covalent_edge_attr,
                                       batch['protein'].res_res_BB_dihs,
                                       batch['protein'].noncovalent_edge_index,
                                       batch['protein'].noncovalent_edge_attr,
                                       distmat_top=None,
                                       distmat_3d= batch['protein'].centriods_dists,
                                       batch=batch['protein'].batch,
                                       return_dense=True)
        
        h_ligand_one, nomask_one = self.forward_ligand_model(batch,'_one',h_prot)
        h_ligand_two, nomask_two = self.forward_ligand_model(batch,'_two',h_prot)
        
        return h_ligand_one, nomask_one, h_ligand_two, nomask_two
    