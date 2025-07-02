def get_config(data_type=1):
    n_layers_p=5
    n_head_p=4
    n_layers_l=5
    n_head_l=4
    lr=1e-5
    v='v0.3'

    config={'n_cores': 24,
            'filters': [(True,'CL1'),(False,'covalent')],
            'split_type': 'leakproof', #core #given pdbs
            'leakproof': 'Data/LP_PDBBind.csv',
            'test_pdbs': [],
            'val_split_type': 'leave_protein_out',
            'max_ligands_pairs_per_prot': 5,
            'val_ratio': 0.1,
            'seed': 42,
            'batch_size': 3,
            'use_sampler':True,
            'distance_param_embedding':True,
            'pharmacophore_param_embedding':True,
            'pure_pharma':False,
            'ligand_res_enc': False,
            'full_protein': True,
            'simple_head': False,
            'protein_config':{'hidden_dim': 100, 'sasa_hidden_dim':10, 'mpnn':'GINE', 'nhead': n_head_l, 'num_layers':n_layers_l,'dropout':0.1,
                             'drop_prob':0.0,'attn_dropout':0.1,'hops_list':[5., 6.75, 7.25, 10.], 'pe_dim':20,
                            'gamma': 0.5, 'slope': 0.0, 'emb_add':True, 'rotamers_hidden_dim': 20 },
            'ligand_config':{'hidden_dim': 100, 'sasa_hidden_dim':10, 'mpnn':'GINE', 'nhead': n_head_p, 'num_layers':n_layers_p,'dropout':0.1,
                             'drop_prob':0.0,'attn_dropout':0.1,'hops_list':[2., 5., 10., 15.],'pe_origin_dim':20, 'pe_dim':20,
                            'gamma': 0.5, 'slope': 0.0 },
            'interaction_config':{'hidden_dim': 132, 'sasa_hidden_dim':10,'sasa_in':2, 'mpnn':'GINE', 'num_layers':n_layers_l,'dropout':0.1,
                             'drop_prob':0.0,'attn_dropout':0.1,'hops_list':[1.,2.,3.,1.],'pe_origin_dim':20, 'pe_dim':20,
                            'gamma': 0.5, 'slope': 0.0 },
            'head_config':{'nhead':4,'n_layers':3,'lr':lr,'pred_hidden_dropout':0.1}
        }

    if data_type==1:
        trainingdata_folder='Data/training_data/'
        config['protein_df']='Data/LeakProofWithFiles.csv'
        config['remove_pdbs']=['3bum','3buo','3bux','3buw']
        config['empty_reactions']=['2r1w','4z90','4bps']
        config['ligand_df']='Data/LeakProofWithFiles.csv'
        config['protein_ligand_mapping']=None
        config['processed_protein_folder']=f'{trainingdata_folder}ProteinDataFiles'
        config['processed_ligand_folder']=f'{trainingdata_folder}LeakProofAbsoluteLigandDataFiles'
        config['interactiondata_file']=f'{trainingdata_folder}LeakProof_absolute_interact_file.pt'
        train_name='AbsoluteC1LeakProofSplit'
    else:
        used_val='IC50EC50'
        trainingdata_folder='Data/training_data/'
        config['protein_df']=f'protein_{used_val}_A1_wpairs.csv'
        config['ligand_df']=f'protein_ligands_{used_val}_A1_wpairs.csv'
        config['protein_ligand_mapping']=f'protein_ligand_{used_val}_A1_mapping_wpairs.pt'
        config['processed_protein_folder']=f'{trainingdata_folder}ProteinDataFiles'
        config['processed_ligand_folder']=f'{trainingdata_folder}A1{used_val}AbsoluteLigandDataFiles'
        config['interactiondata_file']=f'{trainingdata_folder}A1_{used_val}_absolute_interact_file.pt'
        train_name='AbsoluteC1LKA1BindingNetSplit'

    s=+config['use_sampler']
    dpe=+config['distance_param_embedding']
    lre=+config['ligand_res_enc']
    sh=+config['simple_head']
    par_e=+config['pharmacophore_param_embedding']
    ph=+config['pure_pharma']
    pm=+config['full_protein']
    train_name=f'{train_name}_s-{s}_phcph-{ph}_pe-{par_e}_dpe-{dpe}_pm-{pm}_lre-{lre}_sh-{sh}_nblp-{n_layers_p}_nbhp-{n_head_p}_nbll-{n_layers_l}_nbnl-{n_head_l}_lr-{lr}_{v}'
    
    return config,train_name
