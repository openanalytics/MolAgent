import pandas as pd
from typing import Any, Dict, List, Optional, TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

from smolagents import tool 




@tool
def training_answer(data_dict: Dict[str,Any]={},file_nm:str=f'model_metrics.json') -> str:
    """
    This tool formats the results from the model evaluation. 

    Args:
        data_dict: dictionary containing the model performance metrics
        file_nm: filename to save the dictionary, must include .json at the end
    """
    import os 
    def sanitize_path(path):
        return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")
    file_nm=sanitize_path(file_nm)
    import json
    def valid_keys(d:Dict[Any,Any]={})->bool:
        for key,item in d.items():
            if not isinstance(key, (bool, int, str)):
                return False
        return True

    def stringify_keys(d:Dict[Any,Any]={})->Dict[str,Any]:
        new_d={}
        for key,item in d.items():
            new_d[str(key)]=item
        return new_d
    
    if valid_keys(data_dict):
        save_d=data_dict
    else:
        save_d = stringify_keys(data_dict)
    
    dict_nm=file_nm
    with open(dict_nm, 'w') as fp:
        json.dump(save_d, fp)

    return f'The model has been trained, the model performance metrics are saved here: {os.path.abspath(dict_nm)}'

@tool
def load_data_for_training(data_file:str='') -> Dict[str,Any]:
    """
    This tool load the training data from the data retrieving team member and returns a dictionary with the data under the key train_data
    Args:
        data_file: the csv file containing the data 
    """

    import os
    import pandas as pd
    data_dict={}
    for key,file in zip(['data'],[data_file]):
        if os.path.exists(data_file):
            data_dict[key]=pd.read_csv(data_file)
        else:
            raise RuntimeError(f'The file {data_file} does not exist')
        
    return data_dict


  
@tool
def prepare_data_for_modeling(df_given: pd.DataFrame, smiles_column: str, property: str, 
                             task: str = 'regression', use_log10: Optional[bool] = False, use_logit: Optional[bool] = False, categorical: Optional[bool] = True, nb_classes: Optional[int] = 2,
                             class_values:Optional[List[float]]=None, use_quantiles:Optional[bool]=False, 
                              min_allowed_class_samples: Optional[int] = 10,percentages: Optional[bool] = False, check_rdkit_desc:Optional[bool]=True) -> Dict[str, Any]:
    """
    Prepares data for modeling by standardizing SMILES, handling property transformations, and splitting data into training and validation sets. You can not do this yourself and this tool must be called before the tool create_validation_split. 
    
    Args:
        df_given: the pandas dataframe containing the training data set
        smiles_column: the column name of the provided pandas dataframes containing the smiles
        property: the column name of the provided pandas dataframes containing the values to be modelled
        task: the name of the modelling task, can be 'Regression', 'Classification', or 'RegressionClassification'
        use_log10: whether to apply log10 transformation to regression properties, only use for positive values for the given property
        use_logit: whether to apply logit transformation to regression properties
        categorical: whether the target values are already divided in to classes for classification
        nb_classes: the number of classes for the target in case of classification
        class_values: A list of thresholds or quantiles to divide a continuous and non-categorical target in separate classes in case of classification
        use_quantiles: wether the values provided in class_values are quantiles or not
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
    
    if property not in df.columns:
        print(f'The given property {property} is not in the given dataframe')
        raise RuntimeError(f'The given property {property} is not in the given dataframe')
    if smiles_column not in df.columns:
        print(f'The given smiles_column {smiles_column} is not in the dataframe')
        raise RuntimeError(f'The given smiles_column {smiles_column} is not in the dataframe')
    if task not in ['Regression', 'Classification', 'RegressionClassification']:
        raise RuntimeError(f'The given task {task} is not supported, must be one of regression, classification, regressionclassification')
    if len(df) < 1:
        raise RuntimeError(f'The given dataframe has no rows')
    if task == 'Regression':
        if use_log10:
            if not np.all(df[property] > 0):
                raise RuntimeError(f'Using log10 transformation on property {property} requires that the property is positive')
        if use_logit:
            if not np.all(df[property] > 0):
                raise RuntimeError(f'Using logit transformation on property {property} requires that the property is positive')


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
    # Clean data
    df.dropna(inplace=True, subset=[standard_smiles_column])
    df.dropna(inplace=True, how='all', subset=properties)
    df.reset_index(drop=True, inplace=True)

    # Handle property transformations based on task
    transformed_data = False
    if task.lower() == 'classification' or task.lower() == 'regressionclassification':
        if not categorical and class_values is None:
            raise RuntimeError(f'The target is non categorical and no threshold or quantiles are given') 
        prop_builder = ClassBuilder(properties=properties, nb_classes=nb_classes, class_values=class_values,
                               categorical=categorical, use_quantiles=use_quantiles,
                               prefix='Class', min_allowed_class_samples=min_allowed_class_samples, verbose=verbose)
        labelnames = prop_builder.labelnames
    else:
        transformed_data = use_log10 or use_logit
        prop_builder = PropertyTransformer(properties, use_log10=use_log10, use_logit=use_logit, percentages=percentages)
        labelnames = None

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
        'data': df,
        'standard_smiles_column': standard_smiles_column,
        'train_properties': train_properties,
        'original_properties': original_props,
        'transformed_data': transformed_data,
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
    from absolute_config import get_config
    config, train_name = get_config(data_type=1)
    
    from importlib_resources import files  
    model_f=str(files('automol.trained_models').joinpath(chkpt_nm))
    
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
def create_validation_split(data_dict: Dict[str, Any], 
                            validation_strategy: str = 'mixed', 
                            use_available_data_split : bool =True,
                           clustering_method: str = 'Bottleneck', test_size: float = 0.25, 
                           random_state: int = 5) -> Dict[str, Any]:
    """
    Creates validation split for model training based on specified strategy and clustering method. You can not do this yourself and this step is called after the tool prepare_data_for_modeling and before train_automol_model. This can only be done once with the same dataset. In case for many evaluations reload the dataset each time.
    
    Args:
        data_dict: Dictionary from prepare_data_for_modeling containing the prepared data
        validation_strategy: defines the strategy to be used for creating the validation data set; options: 'stratified', 'leave_group_out', or 'mixed'
        use_available_data_split: defines wether to use the already available split if available in the data under the column data_split, default true
        clustering_method: method used for clustering similar compounds; options include 'Bottleneck', 'Butina', 'Scaffold'
        test_size: the proportion of data to use for validation
        random_state: random seed for reproducibility
        
    Returns:
        Dictionary with train and validation datasets and split information
    """
    import numpy as np
    from automol.validation import stratified_validation, leave_grp_out_validation, mixed_validation
    from automol.stacking import FeatureGenerationRegressor
    
    df = data_dict['data']
        
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
    
    if 'data_split' in df and use_available_data_split:        
        Train = df[df['data_split']=='train']
        Validation = df[df['data_split']=='valid']
    else:
        try:
            from automol.stacking_util import get_clustering_algorithm

            c_algo=get_clustering_algorithm(clustering=clustering_method,
                                         n_clusters=val_km_groups,
                                         cutoff=val_butina_cutoff,
                                         include_chirality=val_include_chirality,
                                         verbose=0,
                                         random_state=random_state,
                                         feature_generators=feature_generators,
                                         used_features=features)
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
                    categorical_data=categorical, minority_nb=minority_nb,clustering_algorithm=c_algo
                )
            elif validation_strategy.lower() == 'stratified':
                Train, Validation = stratified_validation(
                    df, train_properties, feature_generators=feature_generators, smiles_data=df_smiles,
                    test_size=test_size, clustering=clustering_method,
                    n_clusters=val_km_groups, cutoff=val_butina_cutoff,
                    include_chirality=val_include_chirality,
                    verbose=0, random_state=random_state, minority_nb=minority_nb, clustering_algorithm=c_algo
                )
            else:  # leave_group_out
                Train, Validation = leave_grp_out_validation(
                    df, train_properties,  feature_generators=feature_generators, smiles_data=df_smiles,
                    test_size=test_size, clustering=clustering_method,
                    n_clusters=val_km_groups, cutoff=val_butina_cutoff,
                    include_chirality=val_include_chirality,
                    verbose=0, random_state=random_state, clustering_algorithm=c_algo
                )
                groups_left_out = np.arange(len(Validation))
        except Exception as e:
            raise RuntimeError(F'splitting of the dataset failed, most commonly reindexing error of dataframe due to multiple calls to this function, error message: {str(e)}') 
    
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
                        computational_load: str = 'cheap', cross_val_fold: int = 4,
                        model_nm:Optional[str]='automol.pt' ) -> Dict[str, Any]:
    """
    Trains an AutoMoL model using the provided data and features from the tools prepare_data_for_modeling and create_validation_split. You can not do this yourself. The tools prepare_data_for_modeling and create_validation_split must be called upfront along with get_feature_generators to get feature generators. 
    
    Args:
        data_dict: Dictionary from create_validation_split containing train/validation data
        features: list of feature keys to use in model training
        blender_properties: list of additionaly properties from the training data set to be used as features in the blender, Optional
        computational_load: indicates allowed execution time - 'cheap', 'intermediate', or 'expensive'
        cross_val_fold: number of folds for cross-validation
        model_nm: the name of model to save, default is 'automol.pt', must include .pt at the end
        
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
    blender_properties=[ p_blen for p_blen in blender_properties if p_blen not in train_properties]
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
    from automol.stacking import save_model

    # Save model to file
    save_model_path = model_nm
    save_model(stacked_model, save_model_path,create_reproducability_output=False)
    
    return {
        'model': stacked_model,
        'results_dictionary': results_dictionary,
        'blender_properties':blender_properties,
        'model_str': model_str,
        'model_save_path': save_model_path,
        **data_dict  # Include all previous data
    }


@tool
def evaluate_automol_model(model_dict: Dict[str, Any],
                            pdf_title:Optional[str]='AutoMol Tool Report',
                            pdf_authors:Optional[str]='AutoMol Tool authors',
                            pdf_mails:Optional[str]='AutoMol Tool e-mail',
                            pdf_summary:Optional[str]='AutoMol Tool model summary',
                            pdf_file:Optional[str]='automol_report.pdf') -> Dict[str, Any]:
    """
    Evaluates the trained AutoMoL model on validation data and returns performance metrics.

    The tool returns a dictionary with the metrics 'execution_time', 'MAE','P_Corr' and 'R2' as keys for regression and 'execution_time', 'accuracy', 'recall', 'precision', 'f1_score' and 'AUC_ROC' for classification. The returned dictionary additionally contains 
    the true values using the key y_true and the predicted values under the key y_pred. 
    
    Args:
        model_dict: Dictionary from train_automol_model containing the trained model and related data
        pdf_title: The title of the pdf report of the model training
        pdf_authors: The authors of the report
        pdf_mails: The emails of the authors of the report
        pdf_summary: the summary of the model training
        pdf_file: the name of the file where to save the pdf report, must include .pdf at the end
        
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
            if prop_cliff_dict is not None and prop_cliff_dict:
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

        #print(properties)
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
    figure_types = available_types #[:1] if task.lower() == 'regression' else available_types[:2]
    
    # Get figures from types
    selected_figures = get_figures_from_types(
        plotly_dictionary=plotly_dictionary,
        properties=properties,
        types=figure_types
    )
    
    # Create captions for report
    captions, types, _ = create_figure_captions(selected_figures)
    
    # Generate PDF report
    title = pdf_title
    authors = pdf_authors
    mails = pdf_mails
    summary = pdf_summary
    output_file = pdf_file
    model_param = {'Task': task, 'Properties': properties}
    
    pdf_data = generate_report(
        plotly_dictionary, selected_figures, captions, types,
        title=title, authors=authors, mails=mails, summary=summary,
        model_param=model_param, model_tostr_list=model_dict['model_str'],
        properties=properties, template_file=None, nb_columns=2
    )
    
    # Save PDF report
    with open(output_file, mode="wb") as f:
        f.write(pdf_data)
    
    
    # Prepare return dictionary with metrics
    result = {}
    p = properties[0]  # Using first property for metrics
    
    if task.lower() == 'regression':
        result = {
            'execution_time': results_dictionary[p]['execution time'],
            'MAE': results_dictionary[p][' MAE'],
            'P_Corr': results_dictionary[p][' P_Corr'],
            'R2': results_dictionary[p][' R2'],
            'y_pred': out[f'predicted_{p}'].tolist(),
            'y_true': stacked_model.Validation[f'{p}'][~na].values.tolist(),
            'model_save_path': model_dict['model_save_path'],
            'report_path': output_file
        }
    else:  # Classification
        result = {
            'execution_time': results_dictionary[p]['execution time'],
            'accuracy': results_dictionary[p][' classification report']['accuracy'],
            'recall': results_dictionary[p][' classification report']['weighted avg']['recall'],
            'precision': results_dictionary[p][' classification report']['weighted avg']['precision'],
            'f1_score': results_dictionary[p][' classification report']['weighted avg']['f1-score'],
            'AUC_ROC': results_dictionary[p][' AUC-ROC'],
            'y_pred': out[f'predicted_{p}'].tolist(),
            'y_true': stacked_model.Validation[f'{p}'].values.tolist(),
            'model_save_path': model_dict['model_save_path'],
            'report_path': output_file
        }
    
    return result