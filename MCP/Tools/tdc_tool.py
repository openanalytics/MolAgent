import os
import pandas as pd

from typing import Any, Dict, List, Optional, Union

from smolagents import Tool, tool

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
            
    def get_crispr_dataset(self, dataset_name:str='',  **kwargs)-> Union[Dict[str, Any], None]:
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

@tool
def retrieve_tdc_data(save_dir:str='tdc_data', dataset_name:str='', group:Union[str,None]=None, label_name:Optional[str]='', mode:Optional[str]='')-> Dict[str,Any]: 
    """This tool retrieves the datasets defined by the given name from therapeutic data commons adme data and returns a dictionary with the data set under the key
     data. Do not print out the returned value. The smiles column is in most cases Drug and the target is Y.
     
    Args:
        save_dir: the directory to save the downloaded data
        dataset_name: the name of the data set to be retrieved from therapeutic data commons
        group: the problem group from the therapeutics data commons, choose from: 
            ADME, Tox, HTS, Epitope, QM, Yields, Develop, CRISPROutcome,
            DTI, PPI, DDI, GDA, DrugRes, DrugSyn, PeptideHMC, AntibodyAff,
            MTI, Catalyst, TCREpitopeBinding, TrialOutcome, ProteinPeptide, PerturbOutcome,
            scDTI, MolGen, RetroSyn, Reaction or SBDD.
        label_name: label name to be used with group CRISPROutcome, Develop or QM, when retrieving the data, optional
        mode: mode to be used with group dti, optional, can be max_affinity or mean

    Returns:
        Dictionary containing the dataset under the key data
    """
    tdc=TDCTool(save_dir=save_dir)
    return tdc.get_dataset(dataset_name = dataset_name,group = group,label_name = label_name, mode = mode)
@tool
def retrieve_tdc_groups()-> List[str]:
    """
    Returns a list of the available problems or groups from the therapeutic data commons. 
    """
    return TDCTool().list_dataset_groups()

@tool
def retrieve_tdc_group_datasets(group:str='ADME') -> List[str]:
    """
    Returns a list of the possible dataset names from the therapeutic data commons for the given group or problem

    Args:
        group: the group or problem from the therapeutic data commons
    """
    from tdc.utils.retrieve import retrieve_dataset_names
    grps=group in retrieve_tdc_groups()
    assert group in retrieve_tdc_groups(), f'group is not available please choose from: {grps}'
    return retrieve_dataset_names(group)



# Usage example
if __name__ == "__main__":
    # Initialize the tool
    tdc_tool = TDCTool()
    
    # List available dataset groups
    groups = tdc_tool.list_dataset_groups()
    print(f"Available dataset groups: {groups}")
    
    # List available datasets in the ADME group
    adme_datasets = tdc_tool.list_datasets('ADME')
    print(f"Available ADME datasets: {adme_datasets}")
    
    # Load an ADME dataset
    caco2_data = tdc_tool.get_dataset('Caco2_Wang', group='ADME')
    if caco2_data is not None:
        print(f"Caco2_Wang dataset shape: {caco2_data['data'].shape}")
        print(f"First 5 rows of Caco2_Wang dataset:\n{caco2_data['data'].head()}")
    
    # Load a drug response dataset
    gdsc_data = tdc_tool.get_dataset('GDSC1', group='Drug Response')
    if gdsc_data is not None:
        print(f"GDSC1 dataset shape: {gdsc_data['data'].shape}")
        print(f"GDSC1 tasks: {gdsc_data['tasks']}")
    
    # Load a generation dataset
    moses_data = tdc_tool.get_dataset('MOSES', group='Generation')
    if moses_data is not None:
        print(f"MOSES dataset shape: {moses_data['data'].shape}")
    
    # Load a benchmark
    admet_benchmark = tdc_tool.get_benchmark('ADMET_Group')
    if admet_benchmark is not None:
        print(f"ADMET_Group benchmark datasets: {admet_benchmark.dataset_names}")
        print(f"ADMET_Group benchmark tasks: {admet_benchmark.dataset_names}")