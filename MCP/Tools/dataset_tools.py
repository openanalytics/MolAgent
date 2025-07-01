import pandas as pd
from typing import Any, Dict, List, Optional, TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

from smolagents import tool 
from smolagents.models import MessageRole, Model

@tool
def data_answer(data_dir:str='prepared_data',file_nm:str='data.csv',df:PandasDataFrame=None) -> str:
    """
    This tool formats the results from the different data preparation tools in the correct format. 

    Args:
        data_dir: the directory to save the csv file
        file_nm: the file name to save the dataset, must include .csv at the end
        df: the pandas dataframe containing the data set
    """
    import os
    def sanitize_path(path):
        return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")
    data_dir=sanitize_path(data_dir)
    file_nm=sanitize_path(file_nm)
    os.makedirs(data_dir, exist_ok=True)
    file=f'{data_dir}/{file_nm}'
    df.to_csv(file)
    return f'The data has been prepared, the data file is saved here: data file: {os.path.abspath(file)} '

@tool
def check_valid_smiles(df:PandasDataFrame=None, smiles_column:str='smiles')-> str:
    """
    A tool that verifies if a column contains smiles or not. Returns true if 50% of the column contains valid smiles. Do not try to validate the smiles yourself, use this tool!

    Args:
        df: the pandas dataframe with the data
        smiles_column: the column contain the smiles
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*') 
    def is_valid_smiles(smiles):                                                                                     
      try:                                                                                                         
          mol = Chem.MolFromSmiles(str(smiles))                                                                    
          return mol is not None                                                                                   
      except:                                                                                                      
          return False
          
    smiles_validity = df[smiles_column].apply(is_valid_smiles)                                                              
    return smiles_validity.mean()>0.5                                                           
                                                                                     

@tool
def retrieve_3d_data(sdf_file:str='ligands.sdf',
                     property_key: str = 'pChEMBL',
                     data_dir:str='prepared_data',
                     file_nm:str='data.csv'
                    ) -> str:
    """
    This tool reads the provided sdf file and returns dictionary with the smiles under the column original_smiles, the values to be model under the provided column name in the argument property_key and the pdb namess in the column pdb 
    
    Args:
        sdf_file: contains the path to the sdf file containing the 3d ligand structures
        property_key: the key used containing the value to model in the sdf_file
        data_dir: the directory to save the csv file
        file_nm: the file name to save the dataset, must include .csv at the end
        
    Returns:
        Dictionary containing the dataframe under the key train
    """
    import numpy as np, pandas as pd
    import sys
    from rdkit import Chem
    import itertools
    from typing import List
    
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
    pdb=[]
    original_smiles=[]
    aff_val=[]
    prot_index_d={}
    
    for idx,mol in enumerate(Chem.SDMolSupplier(sdf_file,removeHs=False)):
        pdb_nm=mol.GetProp('pdb')
        pic50=float(retrieve_prop_from_mol(mol, guesses=[property_key], start=property_key,remove_q=False))
        pdb.append(pdb_nm)
        aff_val.append(pic50)
        original_smiles.append(Chem.MolToSmiles(mol))
        if pdb_nm not in prot_index_d:
            prot_index_d[pdb_nm]=[]
        prot_index_d[pdb_nm].append(idx)

    import os 
    def sanitize_path(path):
        return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")
    data_dir=sanitize_path(data_dir)
    file_nm=sanitize_path(file_nm)
    os.makedirs(data_dir, exist_ok=True)
    file=f'{data_dir}/{file_nm}'
    df=pd.DataFrame({'original_smiles': original_smiles, f'{property_key}': aff_val, 'pdb':pdb })
    df.to_csv(file)
    return f'The data has been prepared, the data file is saved here: data file: {file}'



  