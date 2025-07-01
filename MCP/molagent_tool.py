from smolagents import tool, Tool
import pandas as pd
from typing import Any, Dict, List, Optional
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
from automol.stacking import FeatureGenerationRegressor

def add_prompt_to_system_prompt(prompt:str="")->str:
    
    """
    Analyze a prompt for automal model training
    """
    return f"""You are a professional data analyst. I would like you to analyze the data provided by the information in the user prompt, train a automol model, and provide a summary
    regarding the performance metrics.

        Start by retrieving the data and analyse the target. Here are the specific tasks:
        
        1. First, load or retrieve the data using the information from the user prompt. The user could provide a csv file himself or use one of the tools retrieve_3d_data and 
        retrieve_tdc_data. If the user did not provide a column name for smiles, look for columns with the name smiles, original_smiles, structures, drug or columns containing
        these words. Watch out for columns with ID in their name, these are typically no usable, unless specifically mentioned in the user prompt.
        2. Perform data analysis on the target. Determine the skewness of the target. For the remaining columns of available in the dataset, try to compute the correlation with
        the target. Compute the number of unique values of the target.
        3. Clean the pandas dataframe with the smiles, target and any positively correlated data columns from NaNs.
        4. Prepare the data using the prepare_data_for_modeling tool using the information from the user prompt. If the task is not specified in the user prompt, try regression first.
        If the values of the target are positive and the absolute value of the skewness is below 0.3 use log10 transform. If the task is classification, look for information about the
        classes in the user prompt. If threshold values of class quantiles are given for classification, use these. If no class thresholds or quantiles are given and the user prompt says the target is categorical
        or the number of unique values of the target is less than 6, the target is categorical. If the task is classification and no other information is provided and the 
        target is not categorical, divide the target in two classes by using quantile 0.5 in the tool prepare_data_for_modeling.    
        5. Collect the different feature generators you want to test by using the tools get_feature_generators, add_prolif_feature_generator and add_affinity_graph_feature_generator. 
        Use the information from the user prompt. If no information can be found try the default features generators:  Bottleneck, rdkit and fps_1024_2. You can use these as standalone 
        or combine them. If the user_prompt provides information on a sdf file and pdb folder, try out prolif with all interactions as feature generator too, preferably in combination with others. If no 
        specified information is provided in the user prompt, try out up to 5 different combinations of features used to train a model. 
        6. Split the data in training and validation data using the tool create_validation_split. Only do this once! Look for information in the user prompt about the creation of the validation set. If no
        information is provided and the tool retrieve_tdc_data is used before and validation data is already available, provide the validation data to the tool create_validation_split. 
        Else use the mixed strategy first, if this fails try the stratified one. 
        7. Train a model using the tool train_automol_model for the different feature combinations. Use positively correlated columns to the target from the data set as blender_properties. Look for
        information in the user prompt regarding computational load, if not information is found use the cheapest. 
        8. Evaluate the model using the tool evaluate_automol_model. Collect the different metrics for all the different feature combinations or options.
        9. Based on all this information, provide your professional analysis, highlighting:
           - The features and performance
           - Key technical indicators and what they suggest
           - Create fancy plots: show scatterplots with true values on the x-axis and predicted values for y-axis for regression tasks and confusion matrices for classification tasks. Add the different metrics to 
            the title or in the title
        
        Please organize your response in a clear, structured format suitable for a professional data scientist."""


@tool
def retrieve_tdc_data(data_name:str='Caco2_Wang') -> Dict[str,PandasDataFrame]:
    """This tool retrieves the datasets defined by the given name from therapeutic data commons adme data and returns a dictionary with the data set under the key
     train and the validation set under the key 'valid'. Do not print out the returned value. The smiles column is Drug and the target is Y.
    Args:
        data_name: the name of the data set to be retrieved from therapeutic data commons
    """
    from tdc.single_pred import ADME
    data = ADME(name = data_name)
    split = data.get_split()
    return split

@tool
def retrieve_3d_data(sdf_file:str='ligands.sdf',
                     property_key: str = 'pChEMBL',
                    ) -> Dict[str,PandasDataFrame]:
    """
    This tool reads the provided sdf file and returns dictionary with the smiles under the column original_smiles, the values to be model under the provided column name in the argument property_key and the pdb namess in the column pdb 
    
    Args:
        sdf_file: contains the path to the sdf file containing the 3d ligand structures
        property_key: the key used containing the value to model in the sdf_file
        
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
        
    return {'train': pd.DataFrame({'original_smiles': original_smiles, f'{property_key}': aff_val, 'pdb':pdb })}

    
@tool
def prepare_data_for_modeling(df_given: pd.DataFrame, smiles_column: str, property: str, validation_df: Optional[pd.DataFrame] = None, 
                             task: str = 'regression', use_log10: Optional[bool] = False, use_logit: Optional[bool] = False, categorical: Optional[bool] = True, nb_classes: Optional[int] = 2,
                             class_values:Optional[List[float]]=None, min_allowed_class_samples: Optional[int] = 10,percentages: Optional[bool] = False, check_rdkit_desc:Optional[bool]=True) -> Dict[str, Any]:
    """
    Prepares data for modeling by standardizing SMILES, handling property transformations, and splitting data into training and validation sets. You can not do this yourself and this tool must be called before the tool create_validation_split. 
    
    Args:
        df_given: the pandas dataframe containing the training data set
        smiles_column: the column name of the provided pandas dataframes containing the smiles
        property: the column name of the provided pandas dataframes containing the values to be modelled
        validation_df: the pandas dataframe containing the validation data set (optional)
        task: the name of the modelling task, can be 'Regression', 'Classification', or 'RegressionClassification'
        use_log10: whether to apply log10 transformation to regression properties, only use for positive values for the given property
        use_logit: whether to apply logit transformation to regression properties
        categorical: whether the target values are already divided in to classes for classification
        nb_classes: the number of classes for the targer in case of classification
        class_values: A list of thresholds to divide a continuous and non-categorical target in separate classes in case of classification 
        min_allowed_class_samples: minimum  number of samples per class
        percentages: whether to divide the target by 100 in case the regression property is measured in percentages
        check_rdkit_desc: boolean to remove non valid smiles for rdkit descriptors, default true, Optional
    Returns:
        Dictionary containing the prepared datasets and transformation information
    """
    import warnings
    import sys, os
    import numpy as np
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
    
    from automol.property_prep import validate_rdkit_smiles, add_rdkit_standardized_smiles
    from automol.property_prep import PropertyTransformer, ClassBuilder
    
    df = df_given.copy()
    
    properties = [property]
    standard_smiles_column = 'standardized_smiles'
    verbose = 0
    nb_classes=[nb_classes]

    from rdkit import Chem
    try:
        smi=df[smiles_column][0]
        mol=Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f'creating mol object from first item of smiles_column failed, does smiles_column contain smiles?') 
    except:
        raise RuntimeError(f'creating mol object from first item of smiles_column failed, does smiles_column contain smiles?') 


            
    
    # Standardize SMILES
    add_rdkit_standardized_smiles(df, smiles_column, verbose=verbose, outname=standard_smiles_column)
    if check_rdkit_desc: 
        validate_rdkit_smiles(df, standard_smiles_column,verbose=verbose)
    if validation_df is not None:
        add_rdkit_standardized_smiles(validation_df, smiles_column, verbose=verbose, outname=standard_smiles_column)
        if check_rdkit_desc: 
            validate_rdkit_smiles(df, standard_smiles_column,verbose=verbose)
        
    # Clean data
    df.dropna(inplace=True, subset=[standard_smiles_column])
    df.dropna(inplace=True, how='all', subset=properties)
    df.reset_index(drop=True, inplace=True)

    if validation_df is not None:
            # Clean data
        validation_df.dropna(inplace=True, subset=[standard_smiles_column])
        validation_df.dropna(inplace=True, how='all', subset=properties)
        validation_df.reset_index(drop=True, inplace=True)
    
    # Handle property transformations based on task
    transformed_data = False
    if task.lower() == 'classification' or task.lower() == 'regressionclassification':
        prop_builder = ClassBuilder(properties=properties, nb_classes=nb_classes, class_values=class_values,
                               categorical=categorical, use_quantiles=False,
                               prefix='Class', min_allowed_class_samples=min_allowed_class_samples, verbose=verbose)
        if validation_df is not None:
            validation_df, _ = prop_builder.generate_train_properties(validation_df)
        labelnames = prop_builder.labelnames
    else:
        transformed_data = use_log10 or use_logit
        prop_builder = PropertyTransformer(properties, use_log10=use_log10, use_logit=use_logit, percentages=percentages)
        if validation_df is not None:
            prop_builder_test = PropertyTransformer(properties, use_log10=use_log10, use_logit=use_logit, percentages=percentages)
            prop_builder_test.check_properties(validation_df)
            validation_df, _ = prop_builder_test.generate_train_properties(validation_df)
        labelnames = None

    if validation_df is not None:
        if len(validation_df)==0:
            if use_log10:
                raise RuntimeError(f'empty validation df after data preparation, only use_log10 if the values in {properties[0]} are positive') 
            else:
                raise RuntimeError(F'empty validation df after data preparation') 

    prop_builder.check_properties(df)
    df, train_properties = prop_builder.generate_train_properties(df)

    if len(df)==0:
        if use_log10:
            raise RuntimeError(f'empty df after data preparation, only use_log10 if the values in {properties[0]} are positive') 
        else:
            raise RuntimeError(F'empty df after data preparation') 
    
    original_props = properties
    if transformed_data:
        original_props = prop_builder.original_props

    return {
        'train_data': df,
        'validation_data': validation_df,
        'standard_smiles_column': standard_smiles_column,
        'train_properties': train_properties,
        'original_properties': original_props,
        'transformed_data': transformed_data,
        'property_builder': prop_builder,
        'labelnames': labelnames,
        'task': task,
        'categorical': categorical
    }

@tool
def get_feature_generators(data_dict: Dict[str, Any], feature_key :str ='bottleneck') -> Dict[str, Any]:
    """
    Creates or adds a feature generator to the provided data_dict based on the given feature_key. You can not create a feature generator yourself, you have to call this tool to select a feature generator based on a given feature_key.
    This automol method should be called at before train_automol_model.You can call this method multiple times for different using a different feature_key. 
    You can easily accumulated different feature generators before doing further training. This method is called for each feature type. Different combinations of features can be given in a later stage using the tool train_automol_model.
    
    The available case-sensitive feature keys are : Bottleneck, rdkit, fps_1024_2, fps_2024_2, fps_512_2, fps_2024_3, pcqm4mv2_graphormer_base, gin_supervised_edgepred, gin_supervised_infomax, jtvae_zinc_no_kl,cats,gobbi,pmapper,desc3D, electroshape, usrcat, usr,desc2D, atompair-count, topological-count, fcfp-count, ecfp-count, estate, erg, secfp, pattern, rdkit, fcfp, ecfp, avalon, maccs, ChemBERTa-77M-MTR, ChemBERTa-77M-MLM, MolT5 ChemGPT-1.2B and ChemGPT-4.7M
    Args:
        data_dict: Dictionary from an automol tool containing all the relevant automol data
        feature_key: defines the feature_generator to be added to the list of features generators
        
    Returns:
        Dictionary information of the feature generators 
    """
    if 'feature_generators' in data_dict:
        feature_generators= data_dict['feature_generators']
        del data_dict['feature_generators']
    else:
        from automol.feature_generators import retrieve_default_offline_generators
        feature_generators=retrieve_default_offline_generators(radius=2,nbits=1024)

    if not feature_key in feature_generators:
        if feature_key.startswith('fps'):
            from automol.feature_generators import ECFPGenerator
            splits=feature_key.split('_')
            feature_generators[feature_key]=ECFPGenerator(radius=splits[2], nBits =splits[1])
        if feature_key=='ChemGPT-4.7M' or feature_key=='ChemGPT-1.2B':
            from automol.feature_generators import MolfeatPretrainedHFTransformer
            feature_generators[feature_key]=MolfeatPretrainedHFTransformer( kind=feature_key, notation='selfies',dtype=float,max_length=220)
        if feature_key=='ChemBERTa-77M-MTR' or feature_key=='ChemBERTa-77M-MLM'or feature_key=='MolT5':
            from automol.feature_generators import MolfeatPretrainedHFTransformer
            feature_generators[feature_key]=MolfeatPretrainedHFTransformer( kind=feature_key, notation='smiles',dtype=float,max_length=220)
        fpvec_features=['desc2D', 'atompair-count', 'topological-count', 'fcfp-count', 'ecfp-count', 'estate', 'erg', 'secfp', 'pattern', 'fcfp', 'ecfp', 'avalon', 'maccs']
        if feature_key in fpvec_features:
            from automol.feature_generators import MolfeatFPVecTransformer
            feature_generators[feature_key]= MolfeatFPVecTransformer(kind=feature_key, dtype=float)
        fpvec3d_features=['desc3D', 'electroshape', 'usrcat', 'usr']
        if feature_key in fpvec3d_features:
            from automol.feature_generators import Molfeat3DFPVecTransformer
            feature_generators[feature_key]= Molfeat3DFPVecTransformer(kind=feature_key, dtype=float)
        moltrans_features=['mordred', 'scaffoldkeys', 'cats2d', 'cats','gobbi','pmapper']
        if feature_key in moltrans_features:
            from automol.feature_generators import MolfeatMoleculeTransformer
            if feature_key in ['cats','gobbi','pmapper']:
                from molfeat.calc.pharmacophore import Pharmacophore2D
                feature_generators[feature_key]= MolfeatMoleculeTransformer(featurizer=Pharmacophore2D(factory=feature_key), dtype=float)
            else:
                feature_generators[feature_key]= MolfeatMoleculeTransformer(featurizer=feature_key, dtype=float)
        if feature_key in ['gin_supervised_edgepred', 'gin_supervised_infomax', 'jtvae_zinc_no_kl']:
            from automol.feature_generators import MolfeatPretrainedDGLTransformer
            feature_generators[feature_key]= MolfeatPretrainedDGLTransformer(kind=feature_key, dtype=float)
        if feature_key in ['pcqm4mv2_graphormer_base']:
            from automol.feature_generators import MolfeatGraphormerTransformer
            feature_generators[feature_key]= MolfeatGraphormerTransformer(kind=feature_key, dtype=float)

    return {
        'feature_generators':feature_generators,
        **data_dict  # Include all previous data
    }

@tool
def add_affinity_graph_feature_generator(data_dict: Dict[str, Any],
                                         pdb_folder:str = 'pdbs',
                                         sdf_file:str = 'ligands.sdf',
                                         processed_data_folder:str = 'processed_data',
                                         target:str= 'target',
                                         property_key:str = 'pChEMBL',
                                         chkpt_nm:str ='abs_stratified_leakproof.ckpt'
                                         ) -> Dict[str, Any]:
    """
    Adds the affinity graph feature generator to the provided data_dict. You can not create this feature generator yourself, in order to add this features generator, you need protein structures. These protein structures need to be available
    in one folder as files with the extension pdb. The chemical compounds or ligands should also have 3d structure and are required to be provided as an sdf file. In this sdf file, each ligand has a key pdb reference which pdb to be
    used. 
    
    This automol method should be called at before train_automol_model.  
    
    Args:
        data_dict: Dictionary from an automol tool containing all the relevant automol data 
        pdb_folder: contains a directory asd string to the folder containing the pdb files 
        sdf_file: contains the path to the sdf file containing the 3d ligand structures
        processed_data_folder: contains a directory where to store the processed data
        target: the name of the target
        property_key: the key used containing the value to model in the sdf_file
        chkpt_nm: the name of the checkpoint model to be loaded, can be abs_leakproof.ckpt, abs_lk_bn_val.ckpt or abs_stratified_leakproof.ckpt 
    Returns:
        Dictionary with updated feature generators including the affinity graph features using the key AffGraph  
    """

    if 'feature_generators' in data_dict:
        feature_generators= data_dict['feature_generators']
        del data_dict['feature_generators']
    else:
        from automol.feature_generators import retrieve_default_offline_generators
        feature_generators=retrieve_default_offline_generators(radius=2,nbits=1024)
    

    from automol.structurefeatures.GradFormer.interaction_feature_generator import InteractionFeaturesGenerator
    from pkg_resources import resource_filename
    from absolute_config import get_config
    config, train_name = get_config(data_type=1)
    
    model_f=resource_filename(
                        'automol.trained_models',
                        chkpt_nm
                    )
    
    interaction_feature_generator=InteractionFeaturesGenerator(pdb_folder=pdb_folder,
                         sdf_file=sdf_file,
                         processed_data_folder=processed_data_folder,
                         config=config,
                         name=target,
                         properties=[property_key],
                         model_f=model_f,                                                                   
                         nb_features=132)

    feature_generators['AffGraph']= interaction_feature_generator

    return {
        'feature_generators':feature_generators,
        **data_dict  # Include all previous data
    }
    
@tool
def add_prolif_feature_generator(data_dict: Dict[str, Any],
                                 pdb_folder:str = 'pdbs',
                                 sdf_file:str = 'ligands.sdf',
                                 interactions:Optional[List[str]] = ['Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic', 'Cationic', 'CationPi', 'PiCation', 'VdWContact'],
                                ) -> Dict[str, Any]:
    """
    Adds the prolif feature generator to the provided data_dict. You can not create this feature generator yourself, in order to add this features generator, you need protein structures. These protein structures need to be available
    in one folder as files with the extension pdb. The chemical compounds or ligands should also have 3d structure and are required to be provided as an sdf file. In this sdf file, each ligand has a key pdb reference which pdb to be
    used. The list of interactions to be modeled by prolif can be any combination of the following: Hydrophobic, HBDonor, HBAcceptor, PiStacking, Anionic, Cationic, CationPi,
    PiCation, VdWContact. 
    
    This automol method should be called at before train_automol_model.  
    
    Args:
        data_dict: Dictionary from an automol tool containing all the relevant automol data 
        pdb_folder: contains a directory asd string to the folder containing the pdb files 
        sdf_file: contains the path to the sdf file containing the 3d ligand structures
        interactions: list of interactions to be modeled by prolif, can be any of Hydrophobic, HBDonor, HBAcceptor, PiStacking, Anionic, Cationic, CationPi, PiCation, VdWContact. Optional
    Returns:
        Dictionary with updated feature generators including the prolif features using the key prolif  
    """

    if 'feature_generators' in data_dict:
        feature_generators= data_dict['feature_generators']
        del data_dict['feature_generators']
    else:
        from automol.feature_generators import retrieve_default_offline_generators
        feature_generators=retrieve_default_offline_generators(radius=2,nbits=1024)
        
    from automol.structurefeatures.prolif_interactions import ProlifInteractionCountGenerator

    prolif_generator=ProlifInteractionCountGenerator(pdb_folder=pdb_folder,
                         sdf_file=sdf_file,
                         interactions=interactions
                    )

    feature_generators['prolif']= prolif_generator

    return {
        'feature_generators':feature_generators,
        **data_dict  # Include all previous data
    }


@tool
def create_validation_split(data_dict: Dict[str, Any], validation_strategy: str = 'mixed', 
                           clustering_method: str = 'Bottleneck', test_size: float = 0.25, 
                           random_state: int = 5) -> Dict[str, Any]:
    """
    Creates validation split for model training based on specified strategy and clustering method. You can not do this yourself and this step is called after the tool prepare_data_for_modeling and before train_automol_model. This can only be done once with the same dataset. In case for many evaluations reload the dataset each time.
    
    Args:
        data_dict: Dictionary from prepare_data_for_modeling containing the prepared data
        validation_strategy: defines the strategy to be used for creating the validation data set; options: 'stratified', 'leave_group_out', or 'mixed'
        clustering_method: method used for clustering similar compounds; options include 'Bottleneck', 'Butina', 'Scaffold'
        test_size: the proportion of data to use for validation
        random_state: random seed for reproducibility
        
    Returns:
        Dictionary with train and validation datasets and split information
    """
    import numpy as np
    from automol.validation import stratified_validation, leave_grp_out_validation, mixed_validation
    from automol.stacking import FeatureGenerationRegressor
    
    df = data_dict['train_data']
    validation_df = data_dict['validation_data']
    train_properties = data_dict['train_properties']
    standard_smiles_column = data_dict['standard_smiles_column']
    task = data_dict['task']

    if not 'feature_generators' in data_dict:
        data_dict= get_feature_generators(data_dict, feature_key = '')

    feature_generators=data_dict['feature_generators']
    
    # Configure model params for validation
    features = ['Bottleneck']  # Default feature
    
    # Parameters for validation split
    val_km_groups = 20
    val_butina_cutoff = [0.6]
    val_include_chirality = True
    minority_nb = 5
    df_smiles = df[standard_smiles_column]
    categorical =data_dict['categorical']
    
    # Split data based on validation strategy
    groups_left_out = None
    prop_cliff_dict = None
    
    if validation_df is None:
        try:
            if validation_strategy.lower() == 'mixed':
                mix_coef_dict = {'prop_cliffs': 0.3, 'leave_group_out': 0.3, 'stratified': 0.4}
                prop_cliff_butina_th = 0.45
                rel_prop_cliff = 0.5
                
                Train, Validation, groups_left_out, prop_cliff_dict = mixed_validation(
                    df_orig=df, properties=train_properties, feature_generators=feature_generators,
                    standard_smiles_column=standard_smiles_column,
                    prop_cliff_cut=rel_prop_cliff, prop_cliff_butina=prop_cliff_butina_th,
                    test_size=test_size, clustering=clustering_method,
                    n_clusters=val_km_groups, cutoff=val_butina_cutoff,
                    include_chirality=val_include_chirality,
                    verbose=0, random_state=random_state, mix_dict=mix_coef_dict,
                    categorical_data=categorical, minority_nb=minority_nb
                )
            elif validation_strategy.lower() == 'stratified':
                Train, Validation = stratified_validation(
                    df, train_properties, feature_generators=feature_generators, smiles_data=df_smiles,
                    test_size=test_size, clustering=clustering_method,
                    n_clusters=val_km_groups, cutoff=val_butina_cutoff,
                    include_chirality=val_include_chirality,
                    verbose=0, random_state=random_state, minority_nb=minority_nb
                )
            else:  # leave_group_out
                Train, Validation = leave_grp_out_validation(
                    df, train_properties,  feature_generators=feature_generators, smiles_data=df_smiles,
                    test_size=test_size, clustering=clustering_method,
                    n_clusters=val_km_groups, cutoff=val_butina_cutoff,
                    include_chirality=val_include_chirality,
                    verbose=0, random_state=random_state
                )
                groups_left_out = np.arange(len(Validation))
        except:
            raise RuntimeError(F'splitting of the dataset failed, most commonly reindexing error of dataframe due to multiple calls to this function') 
    else:
        Train = df
        Validation = validation_df
    
    return {
        'train': Train,
        'validation': Validation,
        'groups_left_out': groups_left_out,
        'prop_cliff_dict': prop_cliff_dict,
        **data_dict  # Include all previous data
    }

@tool
def train_automol_model(data_dict: Dict[str, Any], features: List[str] = ['Bottleneck'],
                        blender_properties:Optional[List[str]]=[],
                        computational_load: str = 'cheap', cross_val_fold: int = 4) -> Dict[str, Any]:
    """
    Trains an AutoMoL model using the provided data and features from the tools prepare_data_for_modeling and create_validation_split. You can not do this yourself. The tools prepare_data_for_modeling and create_validation_split must be called upfront along with get_feature_generators to get feature generators. 
    
    Args:
        data_dict: Dictionary from create_validation_split containing train/validation data
        features: list of feature keys to use in model training
        blender_properties: list of additionaly properties from the training data set to be used as features in the blender, Optional
        computational_load: indicates allowed execution time - 'cheap', 'intermediate', or 'expensive'
        cross_val_fold: number of folds for cross-validation
        
    Returns:
        Dictionary containing the trained model and related information
    """
    import numpy as np
    from automol.stacking_util import  ModelAndParams   

    
    if 'train' not in data_dict or 'validation' not in data_dict: 
        return f"train and validation keys not in provided data_dict, did you use the tool create_validation_split?"
    if isinstance(features,str):
        features=[features]
    Train = data_dict['train']
    Validation = data_dict['validation']
    train_properties = data_dict['train_properties']
    standard_smiles_column = data_dict['standard_smiles_column']
    task = data_dict['task']
    feature_generators=data_dict['feature_generators']
    n_jobs_dict = {'outer_jobs': 1, 'inner_jobs': 6, 'method_jobs': 2}

    for p_blen in blender_properties:
        assert p_blen in Train, f'Property {p_blen} from blender_properties not available as column in provided dataset under the key train in provided data_dict'
    
    param_Factory = ModelAndParams(
        task=task,
        computional_load=computational_load,
        distribution_defaults=False,
        hyperopt_defaults=None,
        feature_generators=feature_generators,
        use_gpu=False,
        normalizer=True,
        top_normalizer=False,
        random_state=[1, 3, 5, 7, 9],
        red_dim_list=['passthrough'],
        method_list=None,
        blender_list=None,
        model_config=None,
        randomized_iterations=100,
        n_jobs_dict=n_jobs_dict,
        labelnames=data_dict.get('labelnames'),
        use_sample_weight=False,
        local_dim_red=False,
        dim_red_n_components=None,
        verbose=0
    )
    
    stacked_model, prefixes, params_grid, blender_params, paramsearch = param_Factory.get_model_and_params()
    
    
    # Setting model parameters
    use_sample_weight = False
    random_state = 5
    
    # Set scorer based on task
    if task.lower() == 'classification':
        scorer = 'balanced_accuracy'
    else:
        scorer = 'r2'
    
    # Set clustering parameters for CV
    cv_clustering = 'Bottleneck'
    km_groups = 20
    butina_cutoff = [0.2, 0.4, 0.6]
    include_chirality = False
    
    # Place validation and training set inside the model generator
    stacked_model.Validation = Validation
    stacked_model.Train = Train
    stacked_model.smiles = standard_smiles_column
    
    # Perform data clustering for cross-validation
    stacked_model.Data_clustering(
        method=cv_clustering, n_groups=km_groups, 
        cutoff=butina_cutoff, include_chirality=include_chirality,
        random_state=random_state
    )
    
    # Choose appropriate cross-validation split method
    cross_val_split = 'GKF' if task.lower() == 'regression' else 'SKF'
    
    # Adjust folds if necessary
    outer_folds = cross_val_fold
    if outer_folds > km_groups:
        if km_groups > 2:
            outer_folds = km_groups - 1
        else:
            km_groups = outer_folds + 1
    
    # Initialize results dictionary
    results_dictionary = {}
    
    # Train model for each property
    for p in train_properties:
        results_dictionary[p] = {}
        sample_weight = None
        
        # Train model
        stacked_model.search_model(
            df=None, prop=p, smiles=standard_smiles_column,
            params_grid=params_grid,
            paramsearch=paramsearch,
            features=features,
            blender_properties=blender_properties,
            scoring=scorer,
            cv=outer_folds-1, outer_cv_fold=outer_folds, 
            split=cross_val_split,
            use_memory=True,
            plot_validation=False,
            refit=False,
            blender_params=blender_params,
            prefix_dict=prefixes,
            random_state=random_state,
            sample_weight=sample_weight,
            results_dict=results_dictionary
        )
    
    # Display model metrics
    model_str = stacked_model.print_metrics()
    
    # Refit the model
    #transformed_data = data_dict.get('transformed_data', False)
    #for p in train_properties:
    #    sample_train = None
    #    sample_val = None
    #    stacked_model.refit_model(models=p, sample_train=sample_train, sample_val=sample_val, prefix_dict=prefixes)
    
    # Clean the model
    #stacked_model.deep_clean()
    #stacked_model.compute_SD = True
    
    # Save model to file
    #save_model_path = 'automol_model.pt'
    #save_model(stacked_model, save_model_path)
    
    return {
        'model': stacked_model,
        'results_dictionary': results_dictionary,
        'blender_properties':blender_properties,
        'model_str': model_str,
    #    'model_save_path': save_model_path,
        **data_dict  # Include all previous data
    }

@tool
def evaluate_automol_model(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the trained AutoMoL model on validation data and returns performance metrics.

    
    The tool returns a dictionary with the metrics 'execution_time', 'MAE','P_Corr' and 'R2' as keys for regression and 'execution_time', 'accuracy', 'recall', 'precision', 'f1_score' and 'AUC_ROC' for classification. The returned dictionary additionally contains 
    the true values using the key y_true and the predicted values under the key y_pred. 
    
    Args:
        model_dict: Dictionary from train_automol_model containing the trained model and related data
        
    Returns:
        Dictionary with model performance metrics and predictions
    """
    import numpy as np
    from automol.plotly_util import PlotlyDesigner
    from automol.plotly_util import create_figure_captions, get_figures_from_types
    from automol.plotly_util import generate_report, save_result_dictionary_to_json
    
    # Extract data
    stacked_model = model_dict['model']
    results_dictionary = model_dict['results_dictionary']
    train_properties = model_dict['train_properties']
    original_properties = model_dict['original_properties']
    blender_properties = model_dict['blender_properties']
    transformed_data = False #model_dict.get('transformed_data', False)
    task = model_dict['task']
    standard_smiles_column = model_dict['standard_smiles_column']
    prop_cliff_dict = model_dict.get('prop_cliff_dict')
    groups_left_out = model_dict.get('groups_left_out')
    
    # Get validation smiles
    smiles_list = stacked_model.Validation[standard_smiles_column]

    blender_props={}
    na=smiles_list.isna()
    for p_blen in blender_properties:
        blender_props[p_blen]=stacked_model.Validation[p_blen]
        na=np.logical_or(na,blender_props[p_blen].isna())

    smiles_list=smiles_list[~na]
    blender_props={ key:item[~na] for key,item in blender_props.items()}
    
    # Predict values
    out = stacked_model.predict(props=None, smiles=smiles_list,blender_properties_dict=blender_props, compute_SD=True, convert_log10=transformed_data)
    
    # Generate visualization based on task
    x_width = 600
    y_height = 450
    plotly_dictionary = {}
    
    if task.lower() == 'regression':
        # Handle property names for regression
        if transformed_data:
            if prop_cliff_dict is not None:
                original_prop_cliff_dict = prop_cliff_dict.copy()
                prop_cliff_dict = {'_'.join(key.split('_')[1:]):val for key, val in prop_cliff_dict.items()}
            properties = original_properties
        else:
            properties = train_properties
        # Set cutoffs for classification
        min_ytrue = [float(np.nanmin(list(stacked_model.Validation[p][~na].values))) for p in properties]
        max_ytrue = [float(np.nanmax(list(stacked_model.Validation[p][~na].values))) for p in properties]
        cutoff = [float(np.round(min_ytrue[i] + (max_ytrue[i]-min_ytrue[i])/2, 2)) for i, p in enumerate(properties)]
        positive_cls = '<'
        apply_tick_transformation = False

        print(properties)
        # Generate regression plots
        fig_designer = PlotlyDesigner()
        fig_l = fig_designer.show_regression_report(
            properties, out, 
            y_true=[stacked_model.Validation[f'{p}'][~na].values for p in properties],
            prop_cliffs=prop_cliff_dict, leave_grp_out=groups_left_out, 
            fig_size=(x_width, y_height), smiles=list(smiles_list.values),
            apply_tick_transformation=apply_tick_transformation,
            results_dict=results_dictionary
        )
        
        fig_adv = fig_designer.show_additional_regression_report(
            properties, out, 
            y_true=[stacked_model.Validation[f'{p}'][~na].values for p in properties],
            prop_cliffs=prop_cliff_dict, leave_grp_out=groups_left_out, 
            fig_size=(x_width, y_height), smiles=list(smiles_list.values),
            apply_tick_transformation=apply_tick_transformation,
            results_dict=results_dictionary
        )
        
        fig_th = fig_designer.show_reg_cutoff_report(
            properties, out, 
            y_true=[stacked_model.Validation[f'{p}'][~na].values for p in properties],
            fig_size=(x_width, y_height), cutoff=cutoff, good_class=positive_cls,
            smiles=list(smiles_list.values), 
            apply_tick_transformation=apply_tick_transformation,
            results_dict=results_dictionary
        )
        
        plotly_dictionary = {**fig_l, **fig_th, **fig_adv}
        
    else:  # Classification
        properties = train_properties
        labelnames = model_dict.get('labelnames')
        cmap = ['PiYG', 'Blues']
        
        # Generate classification plots
        fig_designer = PlotlyDesigner()
        youden_dict, fig_l = fig_designer.show_classification_report(
            properties, out, 
            y_true=[stacked_model.Validation[f'{p}'].values for p in properties], 
            labelnames=labelnames, cmap=cmap, fig_size=(x_width, y_height),
            results_dict=results_dictionary
        )
        
        fig_adv = fig_designer.show_additional_classification_report(
            properties, out, 
            y_true=[stacked_model.Validation[f'{p}'].values for p in properties], 
            labelnames=labelnames, cmap=cmap, fig_size=(x_width, y_height),
            results_dict=results_dictionary
        )
        
        _, fig_th = fig_designer.show_clf_threshold_report(
            properties, out, 
            y_true=[stacked_model.Validation[f'{p}'].values for p in properties],
            youden_dict=youden_dict, labelnames=labelnames, fig_size=(x_width, y_height),
            results_dict=results_dictionary
        )
        
        plotly_dictionary = {**fig_l, **fig_th, **fig_adv}
    
    # Generate report
    _, _, available_types = create_figure_captions(plotly_dictionary.keys())
    
    # Select appropriate figure types
    figure_types = available_types[:1] if task.lower() == 'regression' else available_types[:2]
    
    # Get figures from types
    selected_figures = get_figures_from_types(
        plotly_dictionary=plotly_dictionary,
        properties=properties,
        types=figure_types
    )
    
    # Create captions for report
    captions, types, _ = create_figure_captions(selected_figures)
    
    # Generate PDF report
    title = 'AutoMoL Report'
    authors = 'AutoMoL Tool'
    mails = 'automol@example.com'
    summary = 'Automated Machine Learning Model Report'
    model_param = {'Task': task, 'Properties': properties}
    
    pdf_data = generate_report(
        plotly_dictionary, selected_figures, captions, types,
        title=title, authors=authors, mails=mails, summary=summary,
        model_param=model_param, model_tostr_list=model_dict['model_str'],
        properties=properties, template_file=None, nb_columns=2
    )
    
    # Save PDF report
    output_file = 'automol_report.pdf'
    with open(output_file, mode="wb") as f:
        f.write(pdf_data)
    
    # Save results to JSON
    json_file = 'automol_results.json'
    save_result_dictionary_to_json(result_dict=results_dictionary, json_file=json_file)
    
    # Prepare return dictionary with metrics
    result = {}
    p = properties[0]  # Using first property for metrics
    
    if task.lower() == 'regression':
        result = {
            'execution_time': results_dictionary[p]['execution time'],
            'MAE': results_dictionary[p][' MAE'],
            'P_Corr': results_dictionary[p][' P_Corr'],
            'R2': results_dictionary[p][' R2'],
            'y_pred': out[f'predicted_{p}'],
            'y_true': stacked_model.Validation[f'{p}'][~na].values,
            'report_path': output_file,
            'json_results_path': json_file
        }
    else:  # Classification
        result = {
            'execution_time': results_dictionary[p]['execution time'],
            'accuracy': results_dictionary[p][' classification report']['accuracy'],
            'recall': results_dictionary[p][' classification report']['weighted avg']['recall'],
            'precision': results_dictionary[p][' classification report']['weighted avg']['precision'],
            'f1_score': results_dictionary[p][' classification report']['weighted avg']['f1-score'],
            'AUC_ROC': results_dictionary[p][' AUC-ROC'],
            'y_pred': out[f'predicted_{p}'],
            'y_true': stacked_model.Validation[f'{p}'].values,
            'report_path': output_file,
            'json_results_path': json_file
        }
    
    return result





@tool
def automol_smiles_modelling_tool(df_given:pd.DataFrame, smiles_column:str, property:str,validation_df:pd.DataFrame=None, task:str='regression',features:List[str]=['Bottleneck'],validation_strategy:str='mixed', computational_load:str='cheap') -> dict:
    """
    This is a tool that trains a model using automol, a pipeline for automated machine learning in drug design. This tool builts a hierarchy for models, called a blender. This blender 
    consist of base estimators and a top estimator combining the output of the base estimators in to one output. 
    
    The data for modelling is provided in a csv file separated by a comma given in the argument file. This csv file should contain a column with SMILES given 
    in the argument smiles_column. This csv file should also contain the target with values on which the model needs to be trained provided in argument property. 
    The specific modeling task should be given in the argument task and a computational workload is given to manage the execution time of this tool. There are 3 
    default features provided: 1) rdkit: shorthand for rdkit descriptors available from the package RDKIT, 2) fps_2024_2: extended connectivity fingerprints (ECFP), namely
    morgan fingerprints from RDKIT using 2024 bits with radius 2, the tool can handle input of fps_<nbits>_<radius> for different values for number of bits and
    radius and lastly 3) Bottleneck: features resulting from a deep learning model representing chemical properties with lower dimension than ECFP. The validation 
    strategy can be chosen too. The given smiles are clustered such that similar compounds are grouped together. The validation set can be defined by providing
    a pandas dataframe in the argument. If no validation dataframe is available, the validation_strategy now defines how the validation
    set is generated: 1) stratified: meaning that each cluster is available in the validation set and the performance metrics are for seen chemistry, 2) leave_group_out:
    meaning that complete clusters are placed in the validation set such that the performance are based on unseen chemistry and 3) mixed: between combination of both
    leave_group_out and stratified providing metrics for both seen and unseen chemistry.  

    The tool returns a dictionary with the metrics 'execution time', 'MAE','P_Corr' and 'R2' as keys for regression and 'execution time', 'accuracy', 'recall', 'precision', 'f1-score' and 'AUC-ROC' for classification. The returned dictionary additionally contains 
    the true values using the key y_true and the predicted value under the key y_pred. 

    Args:
        df_given: the pandas dataframe containing the training data set
        smiles_column: the column name of the provided pandas dataframes containing the smiles
        property: the column name of the provided pandas dataframes containing the values to modelled
        validation_df: the pandas dataframe containing the validation data set
        task: the name of the modelling task, this value should be Regression, RegressionClassification or Classification
        features: a list of features names. These are the features used in model training. The users can select from rdkit meaning rdkit descriptors, fps_2024_2 meaning morgen finger prints and Bottleneck meaning deep learning descriptors representing chemical information
        validation_strategy: defines the strategy to be used for creating the validation data set. Users selects one of three: stratified, leave_group_out or mixed. 
        computational_load: the parameter indicating the allowed execution time of the tool, there are 3 options: cheap, intermediate and expensive. The 3 options are listed by increasing execution time
    """
    import numpy as np
    from  matplotlib import pyplot as plt
    import warnings
    import sys, os
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    from automol.property_prep import validate_rdkit_smiles, add_rdkit_standardized_smiles
    from automol.property_prep import PropertyTransformer,ClassBuilder

    from automol.stacking_util import retrieve_search_options, ModelAndParams, get_clustering_algorithm

    from automol.feature_generators import retrieve_default_offline_generators

    from automol.validation import stratified_validation, leave_grp_out_validation, mixed_validation

    from automol.plotly_util import PlotlyDesigner
    from automol.plotly_util import create_figure_captions, get_figures_from_types
    from automol.plotly_util import generate_report
    from automol.plotly_util import save_result_dictionary_to_json

    from automol.stacking import save_model

    df=df_given.copy()
    model_param={}
    #model_param['Data file']=file

    
    standard_smiles_column='standardized_smiles'
    check_rdkit_desc=False
    verbose=0
    properties=[property]
    categorical=True
    nb_classes=[2]
    min_allowed_class_samples=10
    use_log10=False
    use_logit=False
    percentages=False
    use_sample_weight=False
    use_provided_sample_weight=False
    if task=='Classification':
        scorer='balanced_accuracy'
    else:
        scorer='r2'
    random_state=5
    random_state_list=[1,3,5,7,9]  
    model_config=None
    search_type=None
    use_distributions=False
    red_dim_list_per_prop=None
    red_dim_list=['passthrough']
    local_dim_red=False
    base_list=None
    blender_list=None
    used_features=None
    normalizer=True
    top_normalizer=False
    n_jobs_dict={'outer_jobs': 1,
                'inner_jobs': 6,
                'method_jobs': 2}
    dim_red_n_components=None
    val_clustering='Bottleneck'
    strategy='mixed'
    test_size=0.25
    minority_nb=5
    val_include_chirality=True
    val_butina_cutoff=[0.6]
    val_km_groups=20
    prop_cliff_butina_th=0.45
    rel_prop_cliff=0.5
    mix_coef_dict={ 'prop_cliffs': 0.3,'leave_group_out': 0.3 ,'stratified': 0.4}
    cv_clustering='Bottleneck'
    include_chirality=False
    km_groups=20
    used_features_km=['Bottleneck']
    butina_cutoff=[0.2,0.4,0.6]
    cross_val_split='GKF'
    outer_folds=4
    revert_to_original=False
    weighted_samples_index=None
    select_sample_weights=None
    search_type=None
    randomized_iterations=None
    used_features=[ features for p in properties]    
    cmap=['PiYG','Blues']
    positive_cls='<'
    apply_tick_transformation=False
    cutoff=None
    x_width=600
    y_height=450
    output_file='summary.pdf'
    title='AutoMoL report Title'
    authors='Joris Tavernier'
    mails='joris.tavernier@openanalytics.eu'
    summary='Project details'
    figure_types=None
    nb_columns=2
    save_model_to_file='automol_model.pt'
    json_file='automol_model.json'




    #df= pd.read_csv(file,sep=',', na_values = ['NAN', '?','NaN'])
    add_rdkit_standardized_smiles(df, smiles_column,verbose=verbose,outname=standard_smiles_column)
    if validation_df is not None:
        add_rdkit_standardized_smiles(validation_df, smiles_column,verbose=verbose,outname=standard_smiles_column)

    model_param['Standardization']='rdkit'


    df.dropna(inplace=True, subset = [standard_smiles_column])
    df.dropna(inplace=True,how='all', subset = properties)
    df.reset_index(drop=True,inplace=True)
    model_param['Properties']=properties
    
    if task=='Classification' or task=='RegressionClassification':        
        model_param['Categorical properties?']=categorical
        prop_builder=ClassBuilder(properties=properties,nb_classes=nb_classes,class_values=None,
                                 categorical=categorical,use_quantiles=False,
                                 prefix='Class',min_allowed_class_samples=min_allowed_class_samples,verbose=verbose)
        if validation_df is not None:
            validation_df,_=prop_builder.generate_train_properties(validation_df)
        transformed_data=False
        labelnames=prop_builder.labelnames
        scorer='balanced_accuracy'
        
    else:
        model_param['Log10 transformation?']=use_log10
        model_param['Logit transformation?']=use_logit
        if use_logit: model_param['Divide properties by 100?']=percentages
        transformed_data=use_log10 or use_logit
        prop_builder=PropertyTransformer(properties,use_log10=use_log10,use_logit=use_logit,percentages=percentages)
        if validation_df is not None:
            prop_builder_test=PropertyTransformer(properties,use_log10=use_log10,use_logit=use_logit,percentages=percentages)
            prop_builder_test.check_properties(validation_df)        
            validation_df,_=prop_builder_test.generate_train_properties(validation_df)
        labelnames=None
        
    prop_builder.check_properties(df)        
    df,train_properties=prop_builder.generate_train_properties(df)
    original_props=properties
    if transformed_data:
        original_props=prop_builder.original_props
    df_smiles=df[standard_smiles_column]
    
    if use_sample_weight: model_param['Provided samples to weighted per property']=weighted_samples_index
    if use_sample_weight: model_param['Provided weights per property']=select_sample_weights

    if use_sample_weight and not use_provided_sample_weight:
        df=prop_builder.generate_sample_weights(df,{train_properties[i]:weighted_samples_index[p]  for i,p in enumerate(original_props) },
                                                {train_properties[i]:select_sample_weights[p]  for i,p in enumerate(original_props) })
        
    model_param['Task']=task
    model_param['Score function']=scorer
    model_param['Random state']=random_state
    model_param['Random state list']=random_state_list
    model_param['computational_load']=computational_load

    if model_config is not None: 
        model_param['model_config']=model_config

    if model_config=='single_stack': 
        use_sample_weight=False
    if model_config=='top_stacking':
        use_sample_weight=False
    if model_config=='inner_stacking' :
        use_sample_weight=False

    if search_type is not None:
        randomized_iterations, hyperopt_defaults, distribution_defaults = retrieve_search_options(search_type=search_type,use_distributions=use_distributions, n_iter=randomized_iterations)
    else :
        hyperopt_defaults= None
        distribution_defaults=use_distributions
        randomized_iterations=100
    if search_type is not None: model_param['param search type']=search_type
    model_param['randomized_iterations']=randomized_iterations
    model_param['distribution_defaults']=distribution_defaults
    model_param['hyperopt_defaults']=hyperopt_defaults
    
    if red_dim_list_per_prop is not None:
        assert len(red_dim_list_per_prop)==len(properties), 'provided list of dim reduction methods per property must equal length of list of properties'
        model_param['red_dim_list']=red_dim_list_per_prop
    else:
        model_param['red_dim_list']=red_dim_list
    model_param['local_dim_red']=local_dim_red
    model_param['base_list']=base_list
    model_param['blender_list']=blender_list 
    model_param['used_features']=used_features

    param_Factory=ModelAndParams(task=task,
                                 computional_load=computational_load,
                                 distribution_defaults=distribution_defaults, #use distributions for parameter selection in randomized search
                                 hyperopt_defaults=hyperopt_defaults, #use hyperopt
                                 feature_generators=retrieve_default_offline_generators(radius=2,nbits=2048),
                                 use_gpu=False,
                                 normalizer=normalizer, #normalize input base estimators
                                 top_normalizer=top_normalizer,#normalize output base estimators and thus input top estimator
                                 random_state=random_state_list,#random_state
                                 red_dim_list=red_dim_list,#list of dimensionality reducing methods keys from dim_archive
                                 method_list=base_list,#list of keys from method_archive
                                 blender_list=blender_list,#list of classifier keys from blender_archive
                                 model_config=model_config, #string with the model layout/config
                                 randomized_iterations=randomized_iterations, #number of randomized iterations
                                 n_jobs_dict=n_jobs_dict,
                                 labelnames=labelnames,
                                 use_sample_weight=use_sample_weight,
                                 local_dim_red=local_dim_red,
                                 dim_red_n_components=dim_red_n_components,
                                 verbose=verbose)
    stacked_model, prefixes,params_grid,blender_params,paramsearch = param_Factory.get_model_and_params()
    model_param['Clustering algorithm used for validation']=val_clustering
    model_param['Validation set strategy used']=strategy
    model_param['Ratio of test size wrt to total data size']=test_size
    model_param['minority_nb']=minority_nb
    if val_clustering=="Scaffold" : model_param['Include chiralty?']=val_include_chirality
    if val_clustering=="Butina" or val_clustering=="HierarchicalButina": model_param['Threshold(s) used for butina when creating Validation set']=val_butina_cutoff
    if val_clustering=="Bottleneck" : model_param['Number of clusters for Kmeans++']=val_km_groups

    #split the data in training and validation
    groups_left_out=None
    prop_cliff_dict=None
    if validation_df is None:
        if strategy=='mixed':
            Train, Validation,groups_left_out, prop_cliff_dict = mixed_validation(df_orig=df,properties=prop_builder.properties,stacked_model=stacked_model,standard_smiles_column=standard_smiles_column,
                                                        prop_cliff_cut=rel_prop_cliff,prop_cliff_butina=prop_cliff_butina_th,test_size=test_size,clustering=val_clustering,
                                                        n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                        verbose=verbose,random_state=random_state,mix_dict=mix_coef_dict, categorical_data=categorical,
                                                        minority_nb=minority_nb)

        elif strategy=='stratified':
            Train, Validation = stratified_validation(df,train_properties,stacked_model,df_smiles,test_size=test_size,clustering=val_clustering,
                                                        n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                        verbose=verbose,random_state=random_state,
                                                        minority_nb=minority_nb)

        else:
            Train, Validation = leave_grp_out_validation(df,train_properties,stacked_model,df_smiles,test_size=test_size,clustering=val_clustering,
                                                        n_clusters=val_km_groups,cutoff=val_butina_cutoff,include_chirality=val_include_chirality,
                                                        verbose=verbose,random_state=random_state)
            groups_left_out=np.arange(len(Validation))
    else:
        Train=df
        Validation=validation_df


    #placing the validation and training set inside the model generator
    stacked_model.Validation=Validation
    stacked_model.Train=Train
    stacked_model.smiles=standard_smiles_column # set col smiles name
    
    model_param['Clustering algorithm used for Nested cross-validation']=cv_clustering
    if cv_clustering=="Scaffold" : model_param['Include chiralty for nested cross-validation?']=include_chirality
    if cv_clustering=="Butina" or val_clustering=="HierarchicalButina": model_param['Threshold(s) used for butina when clustering for nested cross-validation']=butina_cutoff
    if cv_clustering=="Bottleneck" : model_param['Number of clusters for Kmeans++ for Nested cross-validation']=km_groups

    #property check
    for p in properties:
        prop_count=df[p].count()
        if prop_count<100:
            print('Warning: property',p,', has less than 100 valid values')
        if cv_clustering=='Bottleneck' and prop_count/km_groups<10:
            print('Warning: on average less than 10 samples per cluster for property',p,', suggested use is to decrease number of groups')

    #clustering the data       
    stacked_model.Data_clustering(method=cv_clustering , n_groups=km_groups,cutoff=butina_cutoff,include_chirality=include_chirality,
                                random_state=random_state)
    

    model_param['Used cross-validation technique']=cross_val_split
    if cross_val_split != 'LGO': model_param['Number of folds']=outer_folds
    #use GKF for regression instead of stratified group kfold
    if task=='Regression' and cross_val_split=='SKF':
        cross_val_split='GKF'

    if outer_folds>km_groups:
        if km_groups>2:
            outer_folds=km_groups-1
        else:
            km_groups=outer_folds+1

    results_dictionary={}
    #empty to put this one first (keep in mind order in dictionary is not reliable)
    for index,p in enumerate(train_properties):
        results_dictionary[p]={}
        sample_weight=None
        if use_sample_weight:
            sample_weight=stacked_model.Train[f'sw_{p}'].values

        features=used_features[index]
        if 'rdkit' in features and not check_rdkit_desc:
            print('Warning, smiles are not validated for nan rdkit descriptors')

        dim_list=None
        if red_dim_list_per_prop is not None:
            dim_list=red_dim_list_per_prop[index]
            if isinstance(dim_list[0],list) and not local_dim_red:
                dim_list=None
            elif isinstance(dim_list[0],list):
                assert len(dim_list)==len(features), 'Number of provided features must equal number of dimensionality reduction method lists in the case of featurewise specific dimensionality reduction / feature selection'       
        params_grid,blender_params = param_Factory.get_feature_params(selected_features=features,dim_list=dim_list)

        stacked_model.search_model(df= None,   prop=p,  smiles=standard_smiles_column,
                                    params_grid=params_grid,
                                    paramsearch=paramsearch,
                                   features=features,
                                  scoring=scorer,
                                  cv=outer_folds-1,  outer_cv_fold=outer_folds, split=cross_val_split, 
                                  use_memory=True,
                                  plot_validation=False, 
                                 refit=False,# no refit with validation. comes later,
                                 blender_params=blender_params
                                  ,prefix_dict=prefixes,random_state=random_state,
                                  sample_weight=sample_weight,
                                  results_dict=results_dictionary)
    model_str=stacked_model.print_metrics()
    
    smiles_list=stacked_model.Validation[standard_smiles_column]
    out=stacked_model.predict( props =None, smiles=smiles_list,compute_SD=True,convert_log10=transformed_data and revert_to_original)
    if task=='Regression':
        if transformed_data and revert_to_original:
            if prop_cliff_dict is not None:
                original_prop_cliff_dict=prop_cliff_dict.copy()
                prop_cliff_dict={'_'.join(key.split('_')[1:]):val for key,val in prop_cliff_dict.items()}
            properties=original_props
        else:
            properties=train_properties
        
        if cutoff is None:
            min_ytrue=[float(np.nanmin(list(stacked_model.Validation[p].values))) for p in properties]
            max_ytrue=[float(np.nanmax(list(stacked_model.Validation[p].values))) for p in properties]
            #set cutoffs for classification
            cutoff=[float(np.round(min_ytrue[i]+ (max_ytrue[i]-min_ytrue[i])/2,2)) for i,p in enumerate(properties)]
    
        fig_l=PlotlyDesigner().show_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],
                                                      prop_cliffs=prop_cliff_dict, leave_grp_out=groups_left_out,fig_size=(x_width,y_height),
                                                      smiles=list(smiles_list.values), apply_tick_transformation=apply_tick_transformation
                                              ,results_dict=results_dictionary)

        fig_adv=PlotlyDesigner().show_additional_regression_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],
                                                                   prop_cliffs=prop_cliff_dict, leave_grp_out=groups_left_out,fig_size=(x_width,y_height),
                                                                   smiles=list(smiles_list.values), apply_tick_transformation=apply_tick_transformation
                                              ,results_dict=results_dictionary)

        fig_th=PlotlyDesigner().show_reg_cutoff_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],
                                                                    fig_size=(x_width,y_height),cutoff=cutoff,good_class=positive_cls,
                                                                smiles=list(smiles_list.values), apply_tick_transformation=apply_tick_transformation
                                              ,results_dict=results_dictionary )
    else: 
        properties=train_properties
        youden_dict,fig_l=PlotlyDesigner().show_classification_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties], labelnames=labelnames,cmap=cmap,fig_size=(x_width,y_height)
                                              ,results_dict=results_dictionary)
        fig_adv=PlotlyDesigner().show_additional_classification_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties], labelnames=labelnames,cmap=cmap,fig_size=(x_width,y_height)
                                              ,results_dict=results_dictionary)
        _,fig_th=PlotlyDesigner().show_clf_threshold_report(properties,out,y_true=[stacked_model.Validation[f'{p}'].values for p in properties],youden_dict=youden_dict, labelnames=labelnames,fig_size=(x_width,y_height)
                                              ,results_dict=results_dictionary)

    plotly_dictionary={**fig_l,**fig_th,**fig_adv}

    _,_,available_types=create_figure_captions(plotly_dictionary.keys())
    
            
    if figure_types is None:
        if task=='Regression':
            figure_types=available_types[:1]
        else:
            figure_types=available_types[:2]
    
    #retrieve figure keys from types ordered per property
    selected_figures=get_figures_from_types(plotly_dictionary=plotly_dictionary,properties=properties,types=figure_types)

    captions,types,_=create_figure_captions(selected_figures)
    pdf_data=generate_report(plotly_dictionary,selected_figures,captions,types,title=title, authors=authors, mails=mails, summary=summary, model_param=model_param,
                             model_tostr_list=model_str, properties=properties, template_file=None, nb_columns=nb_columns )
    with open(output_file, mode="wb") as f:
        f.write(pdf_data)
        
    save_result_dictionary_to_json(result_dict=results_dictionary,json_file=json_file) 
    
    if transformed_data and revert_to_original:
        properties=train_properties

    for p in properties:
        sample_train=None
        sample_val=None
        if use_sample_weight:
            sample_train=stacked_model.Train[f'sw_{p}'].values
            sample_val=stacked_model.Validation[f'sw_{p}'].values
        stacked_model.refit_model(models=p,sample_train=sample_train,sample_val=sample_val,prefix_dict=prefixes)
    ## clean the class first by removing the computed features
    stacked_model.deep_clean()
    stacked_model.compute_SD=True
    save_model(stacked_model, save_model_to_file)

    if task=='Regression':
        return {'execution time': results_dictionary[properties[0]]['execution time'],
                'MAE': results_dictionary[properties[0]][' MAE'],
                'P_Corr': results_dictionary[properties[0]][' P_Corr'],
                'R2': results_dictionary[properties[0]][' R2'],
                'y_pred':out[f'predicted_{properties[0]}'],
                'y_true': Validation[f'{properties[0]}'].values}
    else:
            return {'execution time': results_dictionary[properties[0]]['execution time'],
                'accuracy': results_dictionary[properties[0]][' classification report']['accuracy'],
                'recall': results_dictionary[properties[0]][' classification report']['weighted avg'][' recall'],
                'precision': results_dictionary[properties[0]][' classification report']['weighted avg']['precision'],
                'f1-score': results_dictionary[properties[0]][' classification report']['weighted avg'][' f1-score'],
                'AUC-ROC': results_dictionary[properties[0]]['AUC-ROC'],
                'y_pred':out[f'predicted_{properties[0]}'],
                'y_true': Validation[f'{properties[0]}'].values}

