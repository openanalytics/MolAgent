import os
import pandas as pd

from typing import Any, Dict, List, Optional, Union
from fastmcp import FastMCP

class TDCTool():    
    """
    A wrapper tool for accessing and retrieving datasets from Therapeutics Data Commons (TDC).
    This tool makes it easy to access different types of datasets available on tdcommons.ai.
    """
    
    def __init__(self, save_dir='./tdc_data'):
        """
        Initialize the TDC tool with a specified save directory.
        
        Args:
            save_dir (str): Directory to save downloaded datasets
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def list_dataset_groups(self):
        """
        List all dataset groups available in TDC.
        
        Returns:
            list: List of available dataset groups
        """
        return [
            'ADME', 'Tox', 'HTS', 'Epitope', 'QM', 'Yields', 'Develop', 'CRISPROutcome',
            'DTI', 'PPI', 'DDI', 'GDA', 'DrugRes', 'DrugSyn', 'PeptideHMC', 'AntibodyAff',
            'MTI', 'Catalyst', 'TCREpitopeBinding', 'TrialOutcome', 'ProteinPeptide', 'PerturbOutcome',
            'scDTI', 'MolGen', 'RetroSyn', 'Reaction', 'SBDD'           
        ]
    
    def list_datasets(self, group=None):
        """
        List available datasets, optionally filtered by group.
        
        Args:
            group (str, optional): Filter datasets by group
                Options: 'ADME', 'Toxicity', 'HTS', etc.
        
        Returns:
            list: List of available datasets
        """
        from tdc.utils import retrieve_dataset_names

        return retrieve_dataset_names(group)
    
    def get_adme_dataset(self,  dataset_name:str='', label_name:str='', mode:str='', **kwargs, ) -> Union[Dict[str, Any], None]:
        """
        Retrieve an ADME dataset.
        
        Args:
            dataset_name (str): Name of the ADME dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.single_pred import ADME
            data = ADME(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_single_pred_data(data)
        except Exception as e:
            print(f"Error loading ADME dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_toxicity_dataset(self,  dataset_name:str='', label_name:str='', mode:str='', **kwargs, )-> Union[Dict[str, Any], None]:
        """
        Retrieve a Toxicity dataset.
        
        Args:
            dataset_name (str): Name of the Toxicity dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.single_pred import Tox
            data = Tox(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_single_pred_data(data)
        except Exception as e:
            print(f"Error loading Toxicity dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_hts_dataset(self,  dataset_name:str='', label_name:str='', mode:str='', **kwargs, )-> Union[Dict[str, Any], None]:
        """
        Retrieve a HTS dataset.
        
        Args:
            dataset_name (str): Name of the HTS dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.single_pred import HTS
            data = HTS(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_single_pred_data(data)
        except Exception as e:
            print(f"Error loading HTS dataset '{dataset_name}': {str(e)}")
            return None

    def get_qm_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs, )-> Union[Dict[str, Any], None]:
        """
        Retrieve a QM dataset.
        
        Args:
            dataset_name (str): Name of the QM dataset
            **kwargs: Additional arguments for the dataset loader contains the label_name
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.single_pred import QM
            data = QM(name=dataset_name, label_name=label_name, path=self.save_dir, **kwargs)
            return self._process_single_pred_data(data)
        except Exception as e:
            print(f"Error loading QM dataset '{dataset_name}': {str(e)}")
            return None

    def get_yields_dataset(self,  dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Yields dataset.
        
        Args:
            dataset_name (str): Name of the Yields dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.single_pred import Yields
            data = Yields(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_single_pred_data(data)
        except Exception as e:
            print(f"Error loading Yields dataset '{dataset_name}': {str(e)}")
            return None

    def get_epitope_dataset(self,  dataset_name:str='', label_name:str='', mode:str='',**kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Epitope dataset.
        
        Args:
            dataset_name (str): Name of the Epitope dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.single_pred import Epitope
            data = Epitope(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_single_pred_data(data)
        except Exception as e:
            print(f"Error loading Epitope dataset '{dataset_name}': {str(e)}")
            return None

    def get_develop_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Develop dataset.
        
        Args:
            dataset_name (str): Name of the Develop dataset
            **kwargs: Additional arguments for the dataset loader and the label_name 
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.single_pred import Develop
            data = Develop(name=dataset_name, label_name=label_name, path=self.save_dir, **kwargs)
            return self._process_single_pred_data(data)
        except Exception as e:
            print(f"Error loading develop dataset '{dataset_name}': {str(e)}")
            return None
            
    def get_crispr_dataset(self, dataset_name:str='', label_name:str='',  **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Develop dataset.
        
        Args:
            dataset_name (str): Name of the Develop dataset
            **kwargs: Additional arguments for the dataset loader and the label_name 
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.single_pred import CRISPROutcome
            data = CRISPROutcome(name=dataset_name, label_name=label_name, path=self.save_dir, **kwargs)
            return self._process_single_pred_data(data)
        except Exception as e:
            print(f"Error loading develop dataset '{dataset_name}': {str(e)}")
            return None
            
    def get_dti_dataset(self, dataset_name:str='',mode:str=None, **kwargs) -> Union[Dict[str, Any], None]:
        """
        Retrieve a Drug-Target Interaction dataset.
        
        Args:
            dataset_name (str): Name of the DTI dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import DTI
            data = DTI(name=dataset_name, path=self.save_dir, **kwargs)
            if mode is not None:
                data.harmonize_affinities(mode = mode)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading DTI dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_ddi_dataset(self, dataset_name:str='', label_name:str='', mode:str='',**kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Drug-Drug Interaction dataset.
        
        Args:
            dataset_name (str): Name of the DDI dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import DDI
            data = DDI(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading DDI dataset '{dataset_name}': {str(e)}")
            return None

    def get_ppi_dataset(self, dataset_name:str='', label_name:str='', mode:str='',**kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Protein-Protein Interaction dataset.
        
        Args:
            dataset_name (str): Name of the PPI dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import PPI
            data = PPI(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading PPI dataset '{dataset_name}': {str(e)}")
            return None

    def get_gda_dataset(self, dataset_name:str='', label_name:str='', mode:str='',**kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a GDA dataset.
        
        Args:
            dataset_name (str): Name of the GDA dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import GDA
            data = GDA(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading GDA dataset '{dataset_name}': {str(e)}")
            return None
            
    def get_drug_response_dataset(self, dataset_name:str='', label_name:str='', mode:str='',**kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Drug Response dataset.
        
        Args:
            dataset_name (str): Name of the Drug Response dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import DrugRes
            data = DrugRes(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading Drug Response dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_drug_synergy_dataset(self, dataset_name:str='', label_name:str='', mode:str='',**kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Drug Synergy dataset.
        
        Args:
            dataset_name (str): Name of the Drug Synergy dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import DrugSyn
            data = DrugSyn(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading Drug Synergy dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_PeptideHMC_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a PeptideMHC dataset.
        
        Args:
            dataset_name (str): Name of the PeptideMHC dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import PeptideMHC
            data = PeptideMHC(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading PeptideMHC dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_AntibodyAff_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a AntibodyAff dataset.
        
        Args:
            dataset_name (str): Name of the AntibodyAff dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import AntibodyAff
            data = AntibodyAff(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading AntibodyAff dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_MTI_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a AntibodyAff dataset.
        
        Args:
            dataset_name (str): Name of the AntibodyAff dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import MTI
            data = MTI(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading MTI dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_Catalyst_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Catalyst dataset.
        
        Args:
            dataset_name (str): Name of the Catalyst dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import Catalyst
            data = Catalyst(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading Catalyst dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_TCREpitopeBinding_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a TCREpitopeBinding dataset.
        
        Args:
            dataset_name (str): Name of the TCREpitopeBinding dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import TCREpitopeBinding
            data = TCREpitopeBinding(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading TCREpitopeBinding dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_TrialOutcome_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a TrialOutcome dataset.
        
        Args:
            dataset_name (str): Name of the TrialOutcome dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import TrialOutcome
            data = TrialOutcome(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading TrialOutcome dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_ProteinPeptide_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a ProteinPeptide dataset.
        
        Args:
            dataset_name (str): Name of the ProteinPeptide dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred import ProteinPeptide
            data = ProteinPeptide(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading ProteinPeptide dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_PerturbOutcome_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a PerturbOutcome dataset.
        
        Args:
            dataset_name (str): Name of the PerturbOutcome dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.multi_pred.perturboutcome import PerturbOutcome
            data = PerturbOutcome(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading PerturbOutcome dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_scDTI_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a scDTI dataset.
        
        Args:
            dataset_name (str): Name of the scDTI dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.resource.dataloader import DataLoader
            data = DataLoader(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading scDTI dataset '{dataset_name}': {str(e)}")
            return None

    def get_MolGen_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a MolGen dataset.
        
        Args:
            dataset_name (str): Name of the MolGen dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.generation import MolGen
            data = MolGen(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading MolGen dataset '{dataset_name}': {str(e)}")
            return None

    def get_RetroSyn_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a RetroSyn dataset.
        
        Args:
            dataset_name (str): Name of the RetroSyn dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.generation import RetroSyn
            data = RetroSyn(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading RetroSyn dataset '{dataset_name}': {str(e)}")
            return None

    def get_Reaction_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a Reaction dataset.
        
        Args:
            dataset_name (str): Name of the Reaction dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.generation import Reaction
            data = Reaction(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading Reaction dataset '{dataset_name}': {str(e)}")
            return None

    def get_SBDD_dataset(self, dataset_name:str='', label_name:str='', mode:str='', **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a SBDD dataset.
        
        Args:
            dataset_name (str): Name of the SBDD dataset
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        try:
            from tdc.generation import SBDD
            data = SBDD(name=dataset_name, path=self.save_dir, **kwargs)
            return self._process_multi_pred_data(data)
        except Exception as e:
            print(f"Error loading SBDD dataset '{dataset_name}': {str(e)}")
            return None
    
    def get_dataset(self, dataset_name:str='', group=None, **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a dataset based on its name and group.
        
        Args:
            dataset_name (str): Name of the dataset
            group (str, optional): Group of the dataset
                Options: 'ADME', 'Toxicity', 'Drug Response', etc.
                If None, will try to infer from dataset name
            **kwargs: Additional arguments for the dataset loader
        
        Returns:
            dict: Dictionary containing the dataset and related information
        """
        # If group is not provided, try to infer it
        if group is None:
            # Try using a general data loader to detect the group
            try:
                from tdc.resource.dataloader import DataLoader
                data = DataLoader(name=dataset_name, path=self.save_dir)
                group = data.get_task_type()
            except:
                # If DataLoader doesn't work, search in all groups
                for potential_group in self.list_dataset_groups():
                    if dataset_name in self.list_datasets(potential_group):
                        group = potential_group
                        break
                
                if group is None:
                    print(f"Could not infer group for '{dataset_name}'. Please specify group.")
                    return None
        
        # Map group to the appropriate method
        group_method_map = {
            'ADME': self.get_adme_dataset,
            'Tox': self.get_toxicity_dataset,
            'HTS': self.get_hts_dataset,
            'QM': self.get_qm_dataset,
            'Yields' : self.get_yields_dataset,
            'Epitope': self.get_epitope_dataset,
            'Develop': self.get_develop_dataset,
            'CRISPROutcome': self.get_crispr_dataset,
            'DTI': self.get_dti_dataset,
            'DDI': self.get_ddi_dataset,
            'PPI': self.get_ppi_dataset,
            'GDA': self.get_gda_dataset,
            'Drug Response': self.get_drug_response_dataset,
            'Drug Synergy': self.get_drug_synergy_dataset,
            'PeptideHMC': self.get_PeptideHMC_dataset,
            'AntibodyAff': self.get_AntibodyAff_dataset,
            'MTI': self.get_MTI_dataset,
            'Catalyst': self.get_Catalyst_dataset,
            'TCREpitopeBinding': self.get_TCREpitopeBinding_dataset, 
            'TrialOutcome': self.get_TrialOutcome_dataset,
            'ProteinPeptide': self.get_ProteinPeptide_dataset,
            'PerturbOutcome': self.get_PerturbOutcome_dataset,
            'scDTI': self.get_scDTI_dataset,
            'MolGen': self.get_MolGen_dataset,
            'RetroSyn': self.get_RetroSyn_dataset,
            'Reaction': self.get_Reaction_dataset,
            'SBDD': self.get_SBDD_dataset
        }
        
        # Call the appropriate method based on group
        if group in group_method_map:
            return group_method_map[group](dataset_name, **kwargs)
        else:
            print(f"Invalid group: '{group}'")
            return None
    
    def _process_single_pred_data(self, data)-> Union[Dict[str, Any], None]:
        """
        Process single prediction data into a standardized format.
        
        Args:
            data: TDC single prediction data object
            
        Returns:
            dict: Standardized data dictionary
        """
        splits= data.get_split()
        train=data.get_split()['train']
        train['data_split']=['train']*len(train)
        valid=data.get_split()['valid']
        valid['data_split']=['valid']*len(valid)
        test=data.get_split()['test']
        test['data_split']=['test']*len(test)
        return {
            'data': pd.concat([train,valid,test],axis=0).reset_index(drop=True)
        }
    
    def _process_multi_pred_data(self, data)-> Union[Dict[str, Any], None]:
        """
        Process multiple prediction data into a standardized format.
        
        Args:
            data: TDC multiple prediction data object
            
        Returns:
            dict: Standardized data dictionary
        """
        splits= data.get_split()
        train=data.get_split()['train']
        train['data_split']=['train']*len(train)
        valid=data.get_split()['valid']
        valid['data_split']=['valid']*len(valid)
        test=data.get_split()['test']
        test['data_split']=['test']*len(test)
        return {
            'data': pd.concat([train,valid,test],axis=0).reset_index(drop=True),
            'tasks': data.get_tasks()
        }
    
    def get_benchmark(self, benchmark_name, **kwargs)-> Union[Dict[str, Any], None]:
        """
        Retrieve a benchmark dataset.
        
        Args:
            benchmark_name (str): Name of the benchmark
            **kwargs: Additional arguments for the benchmark loader
        
        Returns:
            object: Benchmark object
        """
        try:
            from tdc import BenchmarkGroup
            benchmark = BenchmarkGroup(name=benchmark_name, path=self.save_dir, **kwargs)
            return benchmark
        except Exception as e:
            print(f"Error loading benchmark '{benchmark_name}': {str(e)}")
            return None


# Create an MCP server instance with the identifier "wiki-summary"
mcp = FastMCP("automol-data")

@mcp.prompt("dataloading")
async def dataset_loading_prompt(text: str) -> list[dict]:
    """Generates a prompt to prepare data"""
    return [
        {"role": "system", "content": """
        You're a helpful agent specializing in cheminformatics and molecular data science.
    You have been submitted a task by your manager who needs prepared data for molecular property prediction.
    
    You are the data scientist in charge of finding, preparing, and validating molecular datasets. Do not split the data! Proceed in the following steps:
    
    ### 1. Data Retrieval
    Choose the appropriate method based on what's provided:
    - **CSV file**: Load the file and create a pandas DataFrame
    - **Therapeutic Data Commons dataset**: Use the provided tools (retrieve_tdc_data, retrieve_tdc_groups, retrieve_tdc_group_datasets) to identify and retrieve the data
    - **SDF file with PDB files**: Use retrieve_3d_data to extract the molecular structures and properties

    ### 2. Data Exploration
    Perform a basic exploratory analysis:
    - Report the dataset dimensions (rows, columns)
    - Identify SMILES and target columns
    - Check for missing values and their percentage
    - Summarize the target variable distribution (min/max/mean/median for regression, class distribution for classification)
    - Report number of unique molecules

    ### 3. SMILES Validation & Molecular Processing
    - Validate SMILES strings using RDKit (report invalid SMILES count)
    - Check for and handle duplicated molecules
    - Standardize SMILES format (canonical SMILES)
    - Calculate basic molecular descriptors if relevant (MW, LogP, TPSA, etc.)
    - Report any molecules that couldn't be processed

    ### 4. Data Cleaning
    - Handle missing values based on context (removal or imputation)
    - Remove invalid SMILES
    - Address outliers in the target variable if appropriate
    - Handle imbalanced data if it's a classification problem

    ### 5. Final Dataset Preparation
    - Ensure the final dataset has at minimum: valid SMILES column, target column(s)
    - Save the cleaned dataset using the tool data_answer
    - Document any significant data transformations applied

    Your final_answer MUST contain these sections:
    ### 1. Task outcome (short version):
    [Concise summary of what was accomplished]

    ### 2. Dataset summary:
    - **Source**: [Where the data came from]
    - **Saved location**: [Path to the saved file], namely the path provided to you by the tools
    - **Size**: [Number of molecules after processing]
    - **SMILES column**: [Column name containing SMILES]
    - **Target column**: [Column name(s) containing prediction targets]
    - **Target distribution**: [Brief statistical summary of target values]

    ### 3. Data quality notes:
    [Report any issues found: invalid SMILES, extreme outliers, class imbalance, etc.]

    ### 4. Additional context:
    [Any other relevant information to help with model building]

    Put all these in your final_answer tool. Everything that you do not pass as an argument to final_answer will be lost. Note that the file path sanitation is part of the tools and you need to return the provided file path from the tools and the path you asked for. 

    Remember that good molecular data preparation is crucial for successful model development. Be thorough in your analysis and clear in your documentation.
    Make sure to include the file with the processed data in your answer to the manager, this is very important.
        """},
        {"role": "user", "content": f"Please prepare a dataset for the following user demands:\n\n{text}"}
    ]


@mcp.tool()
def retrieve_tdc_data(
    save_dir:str='tdc_data',
    dataset_name:str='',
    data_dir:str='prepared_data',
    file_nm:str='data.csv',
    group:Union[str,None]=None,
    label_name:Optional[str]='',
    mode:Optional[str]=''
)-> str: 
    """This tool retrieves the datasets defined by the given name from therapeutic data commons adme data and returns the location of the data file. The data is saved in a sanitized directory, not necessarily the one you provided. Read the returned message to learn the location. The smiles column is in most cases Drug and the target is Y.
     
    Args:
        save_dir: the directory to save the downloaded data
        dataset_name: the name of the data set to be retrieved from therapeutic data commons
        data_dir: the directory to save the csv file
        file_nm: the file name to save the dataset, must include .csv at the end
        group: the problem group from the therapeutics data commons, choose from: 
            ADME, Tox, HTS, Epitope, QM, Yields, Develop, CRISPROutcome,
            DTI, PPI, DDI, GDA, DrugRes, DrugSyn, PeptideHMC, AntibodyAff,
            MTI, Catalyst, TCREpitopeBinding, TrialOutcome, ProteinPeptide, PerturbOutcome,
            scDTI, MolGen, RetroSyn, Reaction or SBDD.
        label_name: label name to be used with group CRISPROutcome, Develop or QM, when retrieving the data, optional
        mode: mode to be used with group dti, optional, can be max_affinity or mean

    Returns:
        message with the location of the saved file. 
    """
    tdc=TDCTool(save_dir=save_dir)
    data_dict=tdc.get_dataset(dataset_name = dataset_name,group = group,label_name = label_name, mode = mode)
    if data_dict is None:
        return f'Dataset {dataset_name} is not available in group {group}, please check available datasets within the group.'
    import os
    def sanitize_path(path):
        return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")
    data_dir=sanitize_path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    file=f'{data_dir}/{file_nm}'
    file=sanitize_path(file)
    data_dict['data'].to_csv(file, index=False)
    if 'tasks' in data_dict:
        t=data_dict['tasks']
        return f'The data has been prepared, the data file is saved here: data file: {os.path.abspath(file)}, the tasks are: {t}'
    else:
        return f'The data has been prepared, the data file is saved here: data file: {os.path.abspath(file)}'


@mcp.tool()
def retrieve_tdc_groups()-> str:
    """
    Returns a list of the available problems or groups from the therapeutic data commons. 
    """
    grp_nms=', '.join(TDCTool().list_dataset_groups())
    return f'The names of the data set groups are: {grp_nms}'

@mcp.tool()
def retrieve_tdc_group_datasets(group:str='ADME') -> str:
    """
    Returns a list of the possible dataset names from the therapeutic data commons for the given group or problem

    Args:
        group: the group or problem from the therapeutic data commons, choose from: 
            ADME, Tox, HTS, Epitope, QM, Yields, Develop, CRISPROutcome,
            DTI, PPI, DDI, GDA, DrugRes, DrugSyn, PeptideHMC, AntibodyAff,
            MTI, Catalyst, TCREpitopeBinding, TrialOutcome, ProteinPeptide, PerturbOutcome,
            scDTI, MolGen, RetroSyn, Reaction or SBDD.
    """
    from tdc.utils.retrieve import retrieve_dataset_names
    grps=', '.join(TDCTool().list_dataset_groups())
    assert group in grps, f'group is not available please choose from: {grps}'
    dataset_nms=', '.join(retrieve_dataset_names(group))
    return f'The names of the datasets in group {group} are: {dataset_nms}'


@mcp.tool()
def retrieve_3d_data(sdf_file:str='ligands.sdf',
                     property_key: str = 'pChEMBL',
                     data_dir:str='prepared_data',
                     file_nm:str='data.csv'
                    ) -> str:
    """
    This tool reads the provided sdf file and returns the location of the data file. The data is saved in a sanitized directory, not necessarily the one you provided. Read the returned message to learn the location. The values to be model under the provided column name in the argument property_key and the pdb names in the column pdb. 
    
    Args:
        sdf_file: contains the path to the sdf file containing the 3d ligand structures
        property_key: the key used containing the value to model in the sdf_file
        data_dir: the directory to save the csv file
        file_nm: the file name to save the dataset, must include .csv at the end
        
    Returns:
        message with the location of the saved file 
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
    os.makedirs(data_dir, exist_ok=True)
    file=f'{data_dir}/{file_nm}'
    file=sanitize_path(file)
    df=pd.DataFrame({'original_smiles': original_smiles, f'{property_key}': aff_val, 'pdb':pdb })
    df.to_csv(file, index=False)
    import time
    time.sleep(60)
    return f'The data has been prepared, the data file is saved here: data file: {os.path.abspath(file)}'

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)

