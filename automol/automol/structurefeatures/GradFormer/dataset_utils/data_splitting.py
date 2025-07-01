import pandas as pd, numpy as np

def leak_proof_split(leak_csv,protein_indices,protein_ids,filters=[(True,'CL3')],add_unk_to_train=False):
    df = pd.read_csv(leak_csv, index_col=0)
    train_mask=df['new_split'] == 'train'
    val_mask= df['new_split'] == 'val'
    test_mask= df['new_split'] == 'test'
    for b,v in filters :
        if b:
            train_mask = train_mask & df[v]
            val_mask = val_mask & df[v]
            test_mask = test_mask & df[v]
        else:
            train_mask = train_mask & ~df[v]
            val_mask = val_mask & ~df[v]
            test_mask = test_mask & ~df[v]
    train_pdbs = df[train_mask].index
    val_pdbs = df[val_mask].index
    test_pdbs = df[test_mask].index
    train_indices=[]
    val_indices=[]
    test_indices=[]
    for ind,pdb in zip(protein_indices,protein_ids):
        if pdb in train_pdbs:
            train_indices.append(ind)
        elif pdb in val_pdbs:
            val_indices.append(ind)
        elif pdb in test_pdbs:
            test_indices.append(ind)
        elif add_unk_to_train:
            train_indices.append(ind)
            
    return train_indices, val_indices, test_indices

def lk_cid_split(leak_csv,ligand_df,chembl_ids,protein_indices,protein_ids,filters=[(True,'CL3')],absolute=True):
    
    df = pd.read_csv(leak_csv, index_col=0)
    train_mask=df['new_split'] == 'train'
    val_mask= df['new_split'] == 'val'
    test_mask= df['new_split'] == 'test'
    for b,v in filters :
        if b:
            train_mask = train_mask & df[v]
            val_mask = val_mask & df[v]
            test_mask = test_mask & df[v]
        else:
            train_mask = train_mask & ~df[v]
            val_mask = val_mask & ~df[v]
            test_mask = test_mask & ~df[v]
    train_pdbs = df[train_mask].index
    val_pdbs = df[val_mask].index
    test_pdbs = df[test_mask].index
    
    df = ligand_df.copy()
    df['chembl_id']=[v.rsplit('_',1)[1][:-4] for v in df["lig_file"]]
    df['new_split']=np.where(df['chembl_id'].isin(chembl_ids), "test", "train")
    train_mask=df['new_split'] == 'train'
    test_mask= df['new_split'] == 'test'

    train_ids = df[train_mask].index
    test_ids = df[test_mask].index
    train_indices=[]
    val_indices=[]
    test_indices=[]
    for ind,pdb in zip(protein_indices,protein_ids):
        pdb_i=ind[0]
        train_j=[]
        test_j=[]
        if pdb in test_pdbs:
            test_indices.append(ind)
        else:
            for lig_j in ind[1]:
                if absolute:
                    if lig_j in train_ids:
                        train_j.append(lig_j)
                    elif lig_j in test_ids:
                        test_j.append(lig_j)
                else:
                    if lig_j[0] in train_ids and lig_j[1] in train_ids:
                        train_j.append(lig_j)
                    elif lig_j[0] in test_ids or lig_j[1] in test_ids:
                        test_j.append(lig_j)
                
            if train_j:
                if pdb in val_pdbs:
                    val_indices.append( (pdb_i,train_j) )
                elif pdb in train_pdbs:
                    train_indices.append( (pdb_i,train_j) )
            if test_j:
                test_indices.append( (pdb_i,test_j) )
            
    return train_indices,val_indices, test_indices



def cid_split(ligand_df,chembl_ids,protein_indices,protein_ids,absolute=True):
    df = ligand_df.copy()
    df['chembl_id']=[v.rsplit('_',1)[1][:-4] for v in df["lig_file"]]
    df['new_split']=np.where(df['chembl_id'].isin(chembl_ids), "test", "train")
    train_mask=df['new_split'] == 'train'
    test_mask= df['new_split'] == 'test'

    train_ids = df[train_mask].index
    test_ids = df[test_mask].index
    
    train_indices=[]
    test_indices=[]
    for ind,pdb in zip(protein_indices,protein_ids):
        pdb_i=ind[0]
        train_j=[]
        test_j=[]
        for lig_j in ind[1]:
            if absolute:
                if lig_j in train_ids:
                    train_j.append(lig_j)
                elif lig_j in test_ids:
                    test_j.append(lig_j)
            else:
                if lig_j[0] in train_ids and lig_j[1] in train_ids:
                    train_j.append(lig_j)
                elif lig_j[0] in test_ids or lig_j[1] in test_ids:
                    test_j.append(lig_j)
        if train_j:
            train_indices.append( (pdb_i,train_j) )
        if test_j:
            test_indices.append( (pdb_i,test_j) )
    
    return train_indices, test_indices

def core_split(leak_csv,protein_indices,protein_ids,filters=[(True,'CL3')],add_unk_to_train=False):
    df = pd.read_csv(leak_csv, index_col=0)
    test_mask=df['category'] == 'core'
    train_mask= ~test_mask
    for b,v in filters :
        if b:
            train_mask = train_mask & df[v]
            test_mask = test_mask & df[v]
        else:
            train_mask = train_mask & ~df[v]
            test_mask = test_mask & ~df[v]
    train_pdbs = df[train_mask].index
    test_pdbs = df[test_mask].index
    train_indices=[]
    test_indices=[]
    for ind,pdb in zip(protein_indices,protein_ids):
        if pdb in train_pdbs:
            train_indices.append(ind)
        elif pdb in test_pdbs:
            test_indices.append(ind)
        elif add_unk_to_train:
            train_indices.append(ind)
    return train_indices,test_indices

fep2_ids= [ '6hvi', '3l9h', '4ui5', '5hnb', '5tbm', '4r1y', '5ehr', '4pv0']
fep1_ids= [ '2qbs', '4djw', '1h1q', '2gmx', '4gih', '4hw3', '2zff', '3fly']


def pdb_id_split(protein_indices,protein_ids,pdb_ids=[]):
    train_indices=[]
    test_indices=[]
    for ind,pdb in zip(protein_indices,protein_ids):
        if pdb in pdb_ids:
            test_indices.append(ind)
        else:
            train_indices.append(ind)
    return train_indices,test_indices
