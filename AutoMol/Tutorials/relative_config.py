data_type=1
n_layers_p=10
n_head_p=5
n_layers_l=10
n_head_l=5
used_val='B2'
lr=1e-5
v='v0.5'

config={'n_cores': 24,
        'filters': [(True,'CL1'),(False,'covalent')],
        'split_type': 'stratified', #core #given pdbs #stratified #leakproof
        'leakproof': 'Data/LP_PDBBind.csv',
        'test_pdbs': [],
        'val_split_type': 'random', #leave_protein_out
        'max_ligands_pairs_per_prot': 5,
        'test_ratio':0.2,
        'val_ratio': 0.2,
        'seed': 42,
        'batch_size': 3,
        'use_sampler':True,
        'distance_param_embedding':True,
        'pharmacophore_param_embedding':True,
        'pure_pharma':False,
        'ligand_res_enc': False,
        'full_protein': False,
        'simple_head': False,
        'protein_config':{'hidden_dim': 100, 'sasa_hidden_dim':10, 'mpnn':'GINE', 'nhead': n_head_l, 'num_layers':n_layers_l,'dropout':0.1,
                         'drop_prob':0.0,'attn_dropout':0.1,'hops_list':[5., 6.75, 7.25, 10.,8.5], 'pe_dim':20,
                        'gamma': 0.5, 'slope': 0.0, 'emb_add':True, 'rotamers_hidden_dim': 20 },
        'ligand_config':{'hidden_dim': 100, 'sasa_hidden_dim':10, 'mpnn':'GINE', 'nhead': n_head_p, 'num_layers':n_layers_p,'dropout':0.1,
                         'drop_prob':0.0,'attn_dropout':0.1,'hops_list':[2., 5., 10., 15.,7.5],'pe_origin_dim':20, 'pe_dim':20,
                        'gamma': 0.5, 'slope': 0.0 },
        'interaction_config':{'hidden_dim': 132, 'sasa_hidden_dim':10,'sasa_in':2, 'mpnn':'GINE', 'nhead': 4, 'num_layers':n_layers_l,'dropout':0.1,
                         'drop_prob':0.0,'attn_dropout':0.1,'hops_list':[1.,2.,3.,1.],'pe_origin_dim':20, 'pe_dim':20,
                        'gamma': 0.5, 'slope': 0.0 },
        'head_config':{'nhead':4,'n_layers':3,'lr':lr,'pred_hidden_dropout':0.1}
    }

trainingdata_folder='Data/training_data/'
config['protein_df']=f'protein_{used_val}_wpairs.csv'
config['ligand_df']=f'protein_ligands_{used_val}_wpairs.csv'
config['protein_ligand_mapping']=f'protein_ligand_{used_val}_mapping_wpairs.pt'
config['processed_protein_folder']=f'{trainingdata_folder}ProteinDataFiles'
config['processed_ligand_folder']=f'{trainingdata_folder}{used_val}RelativeLigandDataFiles'
config['interactiondata_file']=f'{trainingdata_folder}{used_val}_relative_interact_file.pt'
train_name=f'RelativeDecoder{used_val}'

s=+config['use_sampler']
dpe=+config['distance_param_embedding']
lre=+config['ligand_res_enc']
sh=+config['simple_head']
par_e=+config['pharmacophore_param_embedding']
ph=+config['pure_pharma']
pm=+config['full_protein']
split=config['split_type']
train_name=f'{train_name}_s-{s}_phcph-{ph}_pe-{par_e}_dpe-{dpe}_pm-{pm}_lre-{lre}_sh-{sh}_nblp-{n_layers_p}_nbhp-{n_head_p}_nbll-{n_layers_l}_nbnl-{n_head_l}_lr-{lr}_{split}_{v}'
