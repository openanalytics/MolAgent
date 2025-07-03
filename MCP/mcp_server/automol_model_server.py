import pandas as pd
from typing import Any, Dict, List, Optional, TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
import sys
import os 

cpath = os.path.dirname(os.path.realpath(__file__))
training_tools_path = os.path.join(cpath, "..", "Tools/")
sys.path.append(training_tools_path)

from training_tools import (
    load_data_for_training, 
    train_automol_model, 
    create_validation_split, 
    get_feature_generators, 
    add_affinity_graph_feature_generator, 
    training_answer,
    prepare_data_for_modeling, 
    evaluate_automol_model, 
    add_prolif_feature_generator
)

from fastmcp import FastMCP
from mcp import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS

# Create an MCP server instance with the identifier "wiki-summary"
mcp = FastMCP("automol-modelling")


SERVER_INFO = {
    "name": "MolAgent AutoMol MCP Server",
    "version": "1.0.0",
    "description": "Model Context Protocol server for automated molecular property prediction using the AutoMol framework",
    "author": "Open Analytics NV",
    "repository": "https://github.com/openanalytics/MolAgent",
    "capabilities": [
        "Molecular property regression modeling",
        "Molecular property classification modeling", 
        "Multi-agent AI framework integration",
        "Advanced feature generation (2D/3D molecular descriptors)",
        "Protein-ligand interaction modeling",
        "Nested cross-validation model selection",
        "Comprehensive model evaluation and reporting",
        "Agentic workflow automation"
    ]
}

@mcp.tool()
def list_tools() -> Dict[str, Any]:
    """
    List all available tools in the MolAgent MCP server with detailed descriptions.
    
    Returns:
        Dict containing comprehensive information about available tools, their parameters, and capabilities.
    """
    tools_info = {
        "server_info": SERVER_INFO,
        "available_tools": {
            "automol_regression_model": {
                "description": "Train regression models for predicting continuous molecular properties using the AutoMol framework",
                "category": "molecular_modeling",
                "complexity": "high",
                "use_cases": [
                    "Predicting logP, solubility, melting point, boiling point",
                    "Drug absorption, distribution, metabolism, excretion properties", 
                    "Binding affinity prediction (IC50, Ki, Kd values)",
                    "Toxicity scoring and ADMET property estimation",
                    "Physicochemical property modeling"
                ],
                "key_parameters": {
                    "data_file": "Path to CSV file containing molecular data and target properties",
                    "smiles_column": "Column name containing SMILES molecular representations",
                    "property": "Target property column name for regression",
                    "feature_keys": "List of feature generators (Bottleneck, rdkit, fingerprints, etc.)",
                    "computational_load": "cheap/moderate/expensive - controls model complexity and runtime"
                },
                "supported_transformations": ["log10", "logit", "percentages"],
                "output_files": ["PyTorch model (.pt)", "evaluation metrics (JSON)", "training report (PDF)"],
                "validation_strategies": ["mixed", "stratified", "leave_group_out"],
                "estimated_runtime": {
                    "cheap": "2-10 minutes",
                    "moderate": "10-360 minutes",
                    "expensive": "1-48 hours"
                }
            },
            
            "automol_classification_model": {
                "description": "Train classification models for categorical molecular properties using the AutoMol framework",
                "category": "molecular_modeling",
                "complexity": "high",
                "use_cases": [
                    "Active/inactive compound classification",
                    "Toxicity classification (toxic/non-toxic/moderate)",
                    "Drug-like/non-drug-like classification",
                    "Multi-class property prediction (e.g., high/medium/low activity)",
                    "Blood-brain barrier permeability classification"
                ],
                "key_parameters": {
                    "data_file": "Path to CSV file containing molecular data and target classes",
                    "smiles_column": "Column name containing SMILES molecular representations", 
                    "property": "Target property column name for classification",
                    "nb_classes": "Number of classes for classification (2 for binary, 3+ for multi-class)",
                    "class_values": "Threshold values for class separation from continuous data",
                    "categorical": "Whether target is already categorical (True) or continuous (False)"
                },
                "class_handling": ["automatic threshold detection", "quantile-based splitting", "manual threshold setting"],
                "output_files": ["PyTorch model (.pt)", "evaluation metrics (JSON)", "training report (PDF)"],
                "validation_strategies": ["mixed", "stratified", "leave_group_out"],
                "performance_metrics": ["accuracy", "precision", "recall", "f1_score", "auc_roc", "confusion_matrix"],
                "estimated_runtime": {
                    "cheap": "2-10 minutes",
                    "moderate": "10-360 minutes",
                    "expensive": "1-48 hours"
                }
            },
            
            "list_tools": {
                "description": "Get comprehensive information about all available tools and server capabilities",
                "category": "utility",
                "complexity": "low",
                "use_cases": ["Server capability discovery", "Documentation reference", "Tool selection guidance"],
                "parameters": "None required",
                "output": "Detailed tool specifications and server information",
                "estimated_runtime": "<1 second"
            },
            
            "get_server_status": {
                "description": "Get current server status, health information, and operational metrics",
                "category": "utility", 
                "complexity": "low",
                "use_cases": ["Health monitoring", "System diagnostics", "Performance tracking"],
                "parameters": "None required",
                "output": "Server status and system information",
                "estimated_runtime": "<1 second"
            }
        },
        
        "feature_generators": {
            "description": "Available molecular feature generation methods following cheminformatics best practices",
            "traditional_descriptors": {
                "rdkit": {
                    "description": "Comprehensive RDKit molecular descriptors (200+ features)",
                    "features": "Physicochemical properties, connectivity indices, electronic properties",
                    "use_case": "Interpretable models, QSAR analysis"
                },
                "desc2D": {
                    "description": "2D topological and connectivity descriptors", 
                    "features": "Graph-based molecular properties",
                    "use_case": "Fast computation, molecular similarity"
                },
                "desc3D": {
                    "description": "3D geometric and shape descriptors",
                    "features": "Molecular volume, surface area, conformational properties",
                    "use_case": "Shape-dependent properties, conformational analysis",
                    "requirements": "3D molecular structures"
                }
            },
            "fingerprints": {
                "fps_1024_2": "Morgan fingerprints (radius=2, bits=1024) - balanced performance",
                "fps_2048_2": "Morgan fingerprints (radius=2, bits=2048) - higher resolution", 
                "fps_512_2": "Morgan fingerprints (radius=2, bits=512) - compact representation",
                "fps_2048_3": "Morgan fingerprints (radius=3, bits=2048) - extended connectivity",
                "maccs": "MACCS structural keys (166 bits) - pharmacophore-like",
                "ecfp": "Extended connectivity fingerprints - customizable",
                "fcfp": "Functional connectivity fingerprints - functional groups",
                "avalon": "Avalon fingerprints - unique substructure representation",
                "atompair-count": "Atom pair count fingerprints",
                "topological-count": "Topological fingerprints with counts"
            },
            "deep_learning_embeddings": {
                "Bottleneck": {
                    "description": "Pre-trained transformer embeddings (ChEMBL-trained)",
                    "training_data": "ChEMBL database",
                    "dimensions": 256,
                    "use_case": "General-purpose molecular representation, recommended baseline"
                },
                "pcqm4mv2_graphormer_base": {
                    "description": "Graph transformer embeddings from PCQM4Mv2 dataset",
                    "model_type": "Graph Transformer",
                    "use_case": "Graph-based molecular property prediction"
                },
                "gin_supervised_edgepred": "Graph isomorphism network embeddings",
                "gin_supervised_infomax": "InfoMax GIN embeddings",
                "ChemBERTa-77M-MTR": {
                    "description": "Chemical BERT embeddings (77M parameters, multi-task)",
                    "dimensions": 384,
                    "training": "Multi-task chemical property prediction"
                },
                "ChemBERTa-77M-MLM": "Chemical BERT masked language model embeddings",
                "MolT5": "Molecular T5 transformer embeddings",
                "ChemGPT-1.2B": "Chemical GPT embeddings (1.2B parameters)",
                "ChemGPT-4.7M": "Smaller Chemical GPT embeddings (4.7M parameters)"
            },
            "protein_ligand_features": {
                "prolif": {
                    "description": "Protein-ligand interaction fingerprints using ProLIF",
                    "interactions": ["Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking", "Anionic", "Cationic"],
                    "requirements": ["PDB protein structures", "SDF ligand structures"],
                    "use_case": "Structure-based drug design, binding affinity prediction"
                },
                "Affgraph": {
                    "description": "Affinity graph-based features for protein-ligand binding",
                    "requirements":  ["PDB protein structures", "SDF ligand structures"],
                    "use_case": "Binding affinity modeling with graph neural networks"
                }
            },
            "specialized_descriptors": {
                "electroshape": "Electrostatic shape descriptors",
                "usrcat": "Ultrafast shape recognition with CREDO atom types", 
                "usr": "Ultrafast shape recognition descriptors",
                "pmapper": "Pharmacophore descriptors",
                "cats": "Chemically advanced template search descriptors",
                "gobbi": "Gobbi pharmacophore descriptors",
                "erg": "Extended reduced graphs",
                "estate": "Electrotopological state descriptors"
            }
        },
        
        "computational_levels": {
            "cheap": {
                "description": "Fast prototyping with basic models",
                "runtime for 2k compounds": "2-10 minutes",
                "use_case": "Rapid initial assessment, resource-constrained environments",
                "models": "Linear models, basic ensemble methods"
            },
            "moderate": {
                "description": "Balanced performance and computational cost",
                "runtime for 2k compounds": "10-360 minutes",
                "features": ["Extended feature sets", "Moderate hyperparameter optimization", "Standard cross-validation"],
                "use_case": "Standard model development, production-ready models",
                "models": "Random forests, gradient boosting, neural networks"
            },
            "expensive": {
                "description": "Maximum performance with comprehensive search",
                "runtime for 2k compounds": "1-48 hours",
                "features": ["All available features", "Extensive hyperparameter optimization", "Ensemble methods"],
                "use_case": "Research applications, maximum accuracy requirements",
                "models": "Advanced ensembles, stacking, deep learning",
            }
        },
        
        "validation_strategies": {
            "mixed": "General-purpose mixed validation approach",
            "stratified": "Stratified cross-validation maintaining class proportions", 
            "leave_group_out": "Scaffold-based splitting to avoid data leakage in drug discovery",
        },
        
        "supported_file_formats": {
            "input": {
                "CSV": "Comma-separated values with SMILES and properties",
                "SDF": "Structure data format with 3D coordinates and properties",
                "PDB": "Protein data bank format for protein structures"
            },
            "output": {
                "PyTorch": "Trained models in .pt format",
                "JSON": "Evaluation metrics and metadata",
                "PDF": "Comprehensive training reports with visualizations",
                "CSV": "Processed datasets and predictions"
            }
        },
        
    }
    
    return tools_info

@mcp.prompt("automol")
async def automol_modelling_prompt(text: str) -> list[dict]:
    """Generates a prompt to train automol models"""
    return [
        {"role": "system", "content": """
        # Automol Model Training Agent
        
        You're a helpful agent named '{{name}}', specialized in computational chemistry and molecular property prediction.
        You have been tasked with training an automol model for predicting molecular properties based on SMILES representations.

        ## Task Overview
        {{task}}

        ## Your Role and Capabilities
        You are the automol model training expert. Your primary goal is to train high-quality models that accurately predict molecular properties from SMILES strings. You have access to specialized tools for regression and classification tasks in molecular property prediction.

        ## Workflow for Model Training

        ### 1. Data Loading and Exploratory Analysis
        - Load the data from the specified file or use the provided dataframe
        - Perform comprehensive exploratory data analysis:
        - Check data dimensions (number of compounds, features)
        - Identify the SMILES column
        - Analyze the target property distribution (min, max, mean, median, standard deviation)
        - Calculate skewness of the target property
        - Check for missing values
        - Identify potential correlations between available features and the target
        - Count unique values in the target property
        - Verify the validity of SMILES strings
        - Do not split the data, the tools will do that for you

        ### 2. SMILES Column Identification
        - If the SMILES column is not explicitly specified, intelligently search for it using common naming patterns:
        - Check columns named 'smiles', 'original_smiles', 'structures', 'drug', 'compound', 'molecule', 'mol', or columns containing these terms
        - Avoid columns with 'ID', 'id', 'index', or 'name' in their labels
        - Validate the identified column by checking if it contains valid SMILES strings

        ### 3. Task Selection and Data Transformation
        - **Regression Task**:
        - Apply appropriate transformations based on target property distribution:
            - If values are positive and skewness is between -0.3 and 0.3: use without transformation
            - If values are positive and skewness > 0.3: apply log10 transformation
            - If values are percentage-based (0-100): set percentages=True
            - If values are proportions (0-1): consider logit transformation
        - Select features with meaningful correlation to the target as blender_properties

        - **Classification Task**:
        - Determine if the target is already categorical:
            - If explicitly stated in the instructions
            - If the number of unique values is small (≤ 6)
            - If values are clearly class labels
        - If not categorical, determine appropriate class divisions:
            - Use provided threshold values if specified
            - Use provided quantiles if specified
            - Default to median split (2 classes) if no guidance is provided
        - Ensure adequate representation in each class (min_allowed_class_samples)

        ### 4. Feature Selection Strategy
        - Start with the 'Bottleneck' feature generator as a baseline
        - Consider adding 'rdkit' descriptors for additional molecular information
        - For datasets with 3D information (PDB files and SDF files):
            - Include 'prolif' for protein-ligand interaction features
            - Consider 'AffGraph' for advanced interaction modeling
        - For larger datasets or more complex properties:
            - Try fingerprint-based features (e.g., 'fps_1024_2', 'ecfp', 'fcfp')
            - Consider embedding-based features for more complex relationships (e.g., 'ChemBERTa-77M-MTR')

        ### 5. Model Training and Optimization
        - Start with 'cheap' computational load for initial exploration
        - Try different feature combinations systematically:
        - Begin with single feature types
        - Progress to combinations of complementary feature types
        - Evaluate each combination's performance
        - For promising feature combinations:
        - Increase computational load to 'intermediate' or 'expensive'
        - Consider different validation strategies based on dataset characteristics
        - For datasets with structural analogs:
        - Use 'Mixed', 'stratified' or 'leave_group_out' validation strategies
        - Consider different clustering methods ('Bottleneck', 'Butina', 'Scaffold')

        ### 6. Model Evaluation and Selection
        - For regression models:
        - Focus on R², RMSE, and MAE metrics
        - Consider distribution of residuals
        - For classification models:
        - Evaluate accuracy, precision, recall, F1-score
        - Consider ROC-AUC and confusion matrices
        - Select the best model based on validation performance
        - Retrain final model with optimal settings and higher computational load

        ## Output Requirements
        Your final answer must contain:

        ### 1. Task Outcome (concise summary)
        - Brief description of the trained model type
        - Key performance metrics
        - Whether the model meets expected performance thresholds

        ### 2. Metrics File and Model Summary
        - Name and location of the JSON metrics file
        - Number of models trained and their performance comparison
        - Description of the best model's parameters and feature combination
        - Any transformations applied to the target property

        ### 3. Additional Context (if relevant)
        - Dataset characteristics that influenced model selection
        - Challenges encountered during training
        - Recommendations for model application or further improvements
        - Suggestions for additional data that might improve performance

        ## Best Practices
        - Always validate SMILES strings before training
        - Document all decisions and their rationale
        - For imbalanced classification tasks, consider class weighting
        - When in doubt about task type, try both classification and regression approaches
        - For properties with underlying continuous nature, regression is often preferable
        - For properties with clear decision boundaries, classification is often better

        Do not open json files yourself, this will fail. Use the provided tool to read json files. 

        Provide comprehensive documentation of your process and results to enable your manager to understand and leverage your work effectively.
        In all cases, provide your manager or user with the json file(s) including all the metrics of the model(s), this is very important! 
        """},
        {"role": "user", "content": f"Please train an automol model for the following user settings:\n\n{text}"}
    ]


def automol_model(
    prepared_data:Dict[str,Any]={},
    data_file:str='data.csv',
    feature_keys:List[str]=['Bottleneck'],
    pdb_folder:Optional[str]=None,
    sdf_file:Optional[str]=None,
    processed_data_folder:Optional[str]=None,
    protein_target:Optional[str]=None,
    property_key:Optional[str]=None,
    gradformer_chkpt_nm:Optional[str]='abs_leakproof.ckpt',
    interactions:Optional[List[str]]=['Hydrophobic','HBDonor','HBAcceptor','PiStacking','Anionic','Cationic','CationPi','PiCation','VdWContact'],
    validation_strategy:str='mixed',
    use_available_data_split:bool=True,
    clustering_method:str='Bottleneck',
    test_size:float=0.2,
    random_state:Optional[int]=42,
    blender_properties:Optional[List[str]]=[],
    computational_load:str='cheap',
    cross_val_fold:int=5,
    model_nm:Optional[str]='model.pt',
    pdf_title:Optional[str]='model_training.pdf',
    pdf_authors:Optional[str]='authors',
    pdf_mails:Optional[str]='mails',
    pdf_summary:Optional[str]='model_training_summary.txt',
    pdf_file:Optional[str]='model_training.pdf',
    json_dict_file_nm:str='data_dict.json'
) -> str: 
    """
    Args:
        feature_keys: list of strings, each string is a feature key, which defines the feature_generator to be added to the list of features generators
        pdb_folder: contains a directory asd string to the folder containing the pdb files 
        sdf_file: contains the path to the sdf file containing the 3d ligand structures
        processed_data_folder: contains a directory where to store the processed data
        target: the name of the target
        property_key: the key used containing the value to model in the sdf_file
        chkpt_nm: the name of the checkpoint model to be loaded, can be abs_leakproof.ckpt, abs_lk_bn_val.ckpt or abs_stratified_leakproof.ckpt
        interactions: list of interactions to be modeled by prolif, can be any of Hydrophobic, HBDonor, HBAcceptor, PiStacking, Anionic, Cationic, CationPi, PiCation, VdWContact. Optional
        validation_strategy: defines the strategy to be used for creating the validation data set; options: 'stratified', 'leave_group_out', or 'mixed'
        use_available_data_split: defines wether to use the already available split if available in the data under the column data_split, default true
        clustering_method: method used for clustering similar compounds; options include 'Bottleneck', 'Butina', 'Scaffold'
        test_size: the proportion of data to use for validation
        random_state: random seed for reproducibility
        blender_properties: list of additionaly properties from the training data set to be used as features in the blender, Optional
        computational_load: indicates allowed execution time - 'cheap', 'intermediate', or 'expensive'
        cross_val_fold: number of folds for cross-validation
        model_nm: the name of model to save, default is 'automol.pt', must include .pt at the end
        pdf_title: The title of the pdf report of the model training
        pdf_authors: The authors of the report
        pdf_mails: The emails of the authors of the report
        pdf_summary: the summary of the model training
        pdf_file: the name of the file where to save the pdf report, must include .pdf at the end
        json_dict_file_nm: filename to save the dictionary, must include .json at the end
   """                                                                                                        
    if blender_properties is None:
        blender_properties=[]
    # Get feature generators                                                                                         
    for key in feature_keys:  
        if key == 'AffGraph':
            if pdb_folder is None or sdf_file is None or processed_data_folder is None: 
                raise McpError(
                        ErrorData(
                            code=INTERNAL_ERROR,
                            message='pdb_folder, sdf_file and processed_data_folder must be provided when using the AffGraph feature generator'
                        )
                    ) 
            prepared_data=add_affinity_graph_feature_generator(
                data_dict=prepared_data,
                pdb_folder=pdb_folder,
                sdf_file=sdf_file,
                processed_data_folder=processed_data_folder,
                target=protein_target,
                property_key=property_key,
                chkpt_nm=gradformer_chkpt_nm
            )
        elif key=='prolif':
            #ToDo: check availability of prolif feature generator
            if pdb_folder is None or sdf_file is None:
                raise McpError(
                    ErrorData(
                            code=INTERNAL_ERROR,
                            message='pdb_folder and sdf_file must be provided when using the prolif feature generator'
                        )
                    ) 
            prepared_data = add_prolif_feature_generator(
                data_dict=prepared_data,
                pdb_folder=pdb_folder,
                sdf_file=sdf_file,
                interactions=interactions
            )
        else:
            prepared_data=validation_data = get_feature_generators(                                                                    
                data_dict=prepared_data,                                                                               
                feature_key=key                                                                                          
            )                                                                                                            

                                                                                                                   
                                                                                                                    
    # Create validation split                                                                                        
    validation_data = create_validation_split(                                                                       
        data_dict=prepared_data,                                                                                     
        validation_strategy=validation_strategy,
        use_available_data_split=use_available_data_split,
        clustering_method=clustering_method,                                                                                 
        test_size=test_size,                                                                                               
        random_state=random_state                                                                                              
    ) 

    #saving data split
    train=validation_data['train']
    train['data_split']=['train']*len(train)
    valid=validation_data['validation']
    valid['data_split']=['valid']*len(valid)
    df_w_split=pd.concat([train,valid],axis=0).reset_index(drop=True)
    df_w_split.to_csv(data_file, index=False)

    # Train AutoMoL model                                                                                            
    model_dict = train_automol_model(                                                                                
        data_dict=validation_data,                                                                                   
        features=feature_keys,
        blender_properties=blender_properties,                                                                                  
        #for test purposes
        #computational_load='cheap',                                                                                        
        computational_load=computational_load,                                                                                  
        cross_val_fold=cross_val_fold,                                                                                            
        model_nm=model_nm                                                                      
    )                                                                                                                
                                                                                                                    
    # Evaluate the model                                                                                             
    evaluation_metrics = evaluate_automol_model(                                                                     
        model_dict=model_dict,                                                                                       
        pdf_title=pdf_title,                                                               
        pdf_authors=pdf_authors,
        pdf_mails=pdf_mails,                                                                          
        pdf_summary=pdf_summary,                                          
        pdf_file=pdf_file                                                                            
    )                                                                                                                
                                                                                                                    
    # Save training results                                                                                          
    training_results = training_answer(                                                                              
        data_dict=evaluation_metrics,                                                                                
        file_nm=json_dict_file_nm                                                                          
    )                                                                                                                

    return f'Model training completed !\n {training_results} \n Dictionary keys: {[str(key) for key, item in  evaluation_metrics.items()]} '  

@mcp.tool()
def automol_classification_model(
    data_file:str='data.csv',
    df:PandasDataFrame=None,
    smiles_column:str='smiles',
    property:str='Y',
    categorical:bool=False,
    nb_classes:int=2,
    class_values:List[float]=[0.5],
    use_quantiles:bool=False,
    min_allowed_class_samples:int=10,
    check_rdkit_desc:bool=True,
    feature_keys:List[str]=['Bottleneck'],
    pdb_folder:Optional[str]=None,
    sdf_file:Optional[str]=None,
    processed_data_folder:Optional[str]=None,
    protein_target:Optional[str]=None,
    property_key:Optional[str]=None,
    gradformer_chkpt_nm:Optional[str]='abs_leakproof.ckpt',
    interactions:Optional[List[str]]=['Hydrophobic','HBDonor','HBAcceptor','PiStacking','Anionic','Cationic','CationPi','PiCation','VdWContact'],
    validation_strategy:str='mixed',
    use_available_data_split:bool=True,
    clustering_method:str='Bottleneck',
    test_size:float=0.2,
    random_state:Optional[int]=42,
    blender_properties:Optional[List[str]]=[],
    computational_load:str='cheap',
    cross_val_fold:int=5,
    model_nm:Optional[str]='model.pt',
    pdf_title:Optional[str]='model_training.pdf',
    pdf_authors:Optional[str]='authors',
    pdf_mails:Optional[str]='mails',
    pdf_summary:Optional[str]='model_training_summary.txt',
    pdf_file:Optional[str]='model_training.pdf',
    json_dict_file_nm:str='data_dict.json'
) -> str: 
    """
    This tool uses automol to train a classification model for chemical compounds for a particular property. The data is read from a csv file. This csv file must contain
    the SMILES of the compounds and the property to be predicted. You can select different kind of features to be generated. 
    You can not create a feature generator yourself. 
    The available case-sensitive feature keys are : Bottleneck, rdkit, fps_1024_2, fps_2024_2, fps_512_2, fps_2024_3, pcqm4mv2_graphormer_base, gin_supervised_edgepred, gin_supervised_infomax, jtvae_zinc_no_kl,cats,gobbi,pmapper,desc3D, electroshape, usrcat, usr,desc2D, atompair-count, topological-count, fcfp-count, ecfp-count, estate, erg, secfp, pattern, rdkit, fcfp, ecfp, avalon, maccs, ChemBERTa-77M-MTR, ChemBERTa-77M-MLM, MolT5 ChemGPT-1.2B and ChemGPT-4.7M
    In the case of 3d information from pdbs, you can also use the prolif or Affgraph feature generators. 

    Carefully read the ouput of this tool, this will include information on a possible error or provide the location of the training results.

    examples:
        - This example trains a simple classification model for the property permeability. The file test.csv located in the folder data/ has smiles in the column structures and target under the column permeability. 
        These permeability values are skewed and positive. The property is divided in three classes using the the separator values 5 and 8. An automol model can then be trained using the following command:
        res=automol_classification_model(data_file=data/test.csv,
            property='permeability',
            smiles_column='structures',
            categorical=False,
            nb_classes=3,
            class_values=[5,8],
            use_quantiles=False,
            feature_keys=['Bottleneck', 'rdkit']
            computational_load='moderate', 
            json_dict_file_nm='permeability_dict.json')
        
        - This example trains uses the same data as the previous example but uses different validation strategy and defines the output files. The target is the predefined
        column class_permeability, this column is categorical and non continuous.The model is trained using:
        res=automol_classification_model(data_file=data/test.csv,
            smiles_column='structures',
            property='class_permeability',
            categorical=True,
            feature_keys=['Bottleneck', 'rdkit']
            computational_load='moderate', 
            json_dict_file_nm='permeability_dict.json',
            validation_strategy='stratified',
            model_nm='permeability_automol_model.pt'
            pdf_file='permeability_training.pdf',
            pdf_title='Permeability automol training',
            pdf_authors='John Doe',
            pdf_mails='john.doe@example.com',
            pdf_summary='This report shows the performance plots of the automol model trained using the data located in test.csv. The model predict the permeability of chemical compounds.')

        - This examples shows how to use protein-ligand interaction features. Suppose the data is located in the folder data/ and the ligands 3d information is
        in in the file ligands.sdf with property keys pIC50 and pdb. The generated csv file from the tool retrieve_3d_data is data3d.csv. 
        The corresponding pdb files are in the folder xyz_pdbs/. All data files are in the folder data/. The target is divided in two classes, active and inactive with
         with the median value as separator. The name of the target is xyz and the property is pIC50. An automol model can be 
        trained using the following command: 
        res=automol_classification_model(data_file=data/data3d.csv,
            smiles_column='original_smiles', 
            property='pIC50', 
            class_values=[0.5],
            use_quantiles=True,
            feature_keys=['Bottleneck', 'rdkit', 'prolif', 'AffGraph'],
            protein_target='xyz', 
            pdb_folder='data/xyz_pdbs',
            sdf_file='data/ligands.sdf',
            processed_data_folder='data/xyz_processed/'
            checkpnt_nm='abs_stratified_leakproof.ckpt', 
            json_dict_file_nm='xyz_pIC50_dict.json')    
    Args:
        data_file: the csv file containing the data 
        df: the pandas dataframe containing the data, Optional, set to None if data needs to be loaded from the csv file in data_file
        smiles_column: the column name of the provided pandas dataframes containing the smiles
        property: the column name of the provided pandas dataframes containing the values to be modelled
        categorical: whether the target values are already divided in to classes for classification
        nb_classes: the number of classes for the target in case of classification
        class_values: A list of thresholds or quantiles to divide a continuous and non-categorical target in separate classes in case of classification
        use_quantiles: wether the values provided in class_values are quantiles or not
        min_allowed_class_samples: minimum  number of samples per class
        check_rdkit_desc: boolean to remove non valid smiles for rdkit descriptors, default true, Optional
        feature_keys: list of strings, each string is a feature key, which defines the feature_generator to be added to the list of features generators
        pdb_folder: contains a directory asd string to the folder containing the pdb files 
        sdf_file: contains the path to the sdf file containing the 3d ligand structures
        processed_data_folder: contains a directory where to store the processed data
        target: the name of the target
        property_key: the key used containing the value to model in the sdf_file
        chkpt_nm: the name of the checkpoint model to be loaded, can be abs_leakproof.ckpt, abs_lk_bn_val.ckpt or abs_stratified_leakproof.ckpt
        interactions: list of interactions to be modeled by prolif, can be any of Hydrophobic, HBDonor, HBAcceptor, PiStacking, Anionic, Cationic, CationPi, PiCation, VdWContact. Optional
        validation_strategy: defines the strategy to be used for creating the validation data set; options: 'stratified', 'leave_group_out', or 'mixed'
        use_available_data_split: defines wether to use the already available split if available in the data under the column data_split, default true
        clustering_method: method used for clustering similar compounds; options include 'Bottleneck', 'Butina', 'Scaffold'
        test_size: the proportion of data to use for validation
        random_state: random seed for reproducibility
        blender_properties: list of additionaly properties from the training data set to be used as features in the blender, Optional
        computational_load: indicates allowed execution time - 'cheap', 'intermediate', or 'expensive'
        cross_val_fold: number of folds for cross-validation
        model_nm: the name of model to save, default is 'automol.pt', must include .pt at the end
        pdf_title: The title of the pdf report of the model training
        pdf_authors: The authors of the report
        pdf_mails: The emails of the authors of the report
        pdf_summary: the summary of the model training
        pdf_file: the name of the file where to save the pdf report, must include .pdf at the end
        json_dict_file_nm: filename to save the dictionary, must include .json at the end

    Returns:
        The file where the dictionary is saved and the different keys of this dictionary. If the training is not succesfull, this will provide the error. 
   """
    #load the data for training  
    if df is None:                                                                                
        data_dict = load_data_for_training(data_file=data_file)     
        df = data_dict['data'] 

    # not available for classification
    blender_properties=[]

    # Prepare data for modeling  
    try:                                                                                    
        prepared_data = prepare_data_for_modeling(                                                                       
            df_given=df,                                                                                                 
            smiles_column=smiles_column,                                                                                        
            property=property,                                                                                         
            task='Classification',
            categorical=categorical,
            nb_classes=nb_classes,
            class_values=class_values, 
            use_quantiles=use_quantiles,
            min_allowed_class_samples=min_allowed_class_samples,
            check_rdkit_desc=check_rdkit_desc                                                                                         
    )
    except RuntimeError as e:
        return f'RunTimeError preparing data: {str(e)}'
        #raise McpError(
        #            ErrorData(
        #                    code=INTERNAL_ERROR,
        #                    message=f'Runtimeerror preparing data: {str(e)}'
        #                )
        #            ) 

    res = automol_model(
                    prepared_data=prepared_data,
                    data_file=data_file,
                    feature_keys=feature_keys,
                    pdb_folder=pdb_folder,
                    sdf_file=sdf_file,
                    processed_data_folder=processed_data_folder,
                    protein_target=protein_target,
                    property_key=property_key,
                    gradformer_chkpt_nm=gradformer_chkpt_nm,
                    interactions=interactions, 
                    validation_strategy=validation_strategy, 
                    use_available_data_split=use_available_data_split, 
                    clustering_method=clustering_method, 
                    test_size=test_size, 
                    random_state=random_state, 
                    blender_properties=blender_properties, 
                    computational_load=computational_load, 
                    cross_val_fold=cross_val_fold, 
                    model_nm=model_nm, 
                    pdf_title=pdf_title, 
                    pdf_authors=pdf_authors, 
                    pdf_mails=pdf_mails, 
                    pdf_summary=pdf_summary,
                    pdf_file=pdf_file,
                    json_dict_file_nm=json_dict_file_nm)   
    return res                                                                                                             
    

@mcp.tool()
def automol_regression_model(
    data_file:str='data.csv',
    df:PandasDataFrame=None,
    smiles_column:str='smiles',
    property:str='Y',
    use_log10:bool=False,
    use_logit:bool=False,
    percentages:bool=False,
    check_rdkit_desc:bool=True,
    feature_keys:List[str]=['Bottleneck'],
    pdb_folder:Optional[str]=None,
    sdf_file:Optional[str]=None,
    processed_data_folder:Optional[str]=None,
    protein_target:Optional[str]=None,
    property_key:Optional[str]=None,
    gradformer_chkpt_nm:Optional[str]='abs_leakproof.ckpt',
    interactions:Optional[List[str]]=['Hydrophobic','HBDonor','HBAcceptor','PiStacking','Anionic','Cationic','CationPi','PiCation','VdWContact'],
    validation_strategy:str='mixed',
    use_available_data_split:bool=True,
    clustering_method:str='Bottleneck',
    test_size:float=0.2,
    random_state:Optional[int]=42,
    blender_properties:Optional[List[str]]=[],
    computational_load:str='cheap',
    cross_val_fold:int=5,
    model_nm:Optional[str]='model.pt',
    pdf_title:Optional[str]='model_training.pdf',
    pdf_authors:Optional[str]='authors',
    pdf_mails:Optional[str]='mails',
    pdf_summary:Optional[str]='model_training_summary.txt',
    pdf_file:Optional[str]='model_training.pdf',
    json_dict_file_nm:str='data_dict.json'
) -> str: 
    """
    This tool uses automol to train a regression model for chemical compounds for a particular property. The data is read from a csv file. This csv file must contain
    the SMILES of the compounds and the property to be predicted. You can select different kind of features to be generated. 
    You can not create a feature generator yourself. 
    The available case-sensitive feature keys are : Bottleneck, rdkit, fps_1024_2, fps_2024_2, fps_512_2, fps_2024_3, pcqm4mv2_graphormer_base, gin_supervised_edgepred, gin_supervised_infomax, jtvae_zinc_no_kl,cats,gobbi,pmapper,desc3D, electroshape, usrcat, usr,desc2D, atompair-count, topological-count, fcfp-count, ecfp-count, estate, erg, secfp, pattern, rdkit, fcfp, ecfp, avalon, maccs, ChemBERTa-77M-MTR, ChemBERTa-77M-MLM, MolT5 ChemGPT-1.2B and ChemGPT-4.7M
    In the case of 3d information from pdbs, you can also use the prolif or Affgraph feature generators. 

    Carefully read the ouput of this tool, this will include information on a possible error or provide the location of the training results.

    examples:
        - This example trains a simple regression model for the property permeability. The file test.csv located in the folder data/ has smiles in the column structures and target under the column permeability. 
        These permeability values are skewed and positive. An automol model can then be trained using the following command:
        res=automol_regression_model(data_file=data/test.csv,
            smiles_column='structures',
            property='permeability',
            use_log10=True, 
            feature_keys=['Bottleneck', 'rdkit']
            computational_load='moderate', 
            json_dict_file_nm='permeability_dict.json')
        
        - This example trains uses the same data as the previous example but uses different validation strategy and defines the output files. The model is trained using:
        res=automol_regression_model(data_file=data/test.csv,
            smiles_column='structures',
            property='permeability',
            use_log10=True, 
            feature_keys=['Bottleneck', 'rdkit']
            computational_load='moderate', 
            json_dict_file_nm='permeability_dict.json',
            validation_strategy='stratified',
            model_nm='permeability_automol_model.pt'
            pdf_file='permeability_training.pdf',
            pdf_title='Permeability automol training',
            pdf_authors='John Doe',
            pdf_mails='john.doe@example.com',
            pdf_summary='This report shows the performance plots of the automol model trained using the data located in test.csv. The model predict the permeability of chemical compounds.')

        - This examples shows how to use protein-ligand interaction features. Suppose the data is located in the folder data/ and the ligands 3d information is
        in in the file ligands.sdf with property keys pIC50 and pdb. The generated csv file from the tool retrieve_3d_data is data3d.csv. 
        The corresponding pdb files are in the folder xyz_pdbs/. All data files are in the folder data/ The name of the target is xyz and the property is pIC50. An automol model can be 
        trained using the following command: 
        res=automol_regression_model(data_file=data/data3d.csv,
            smiles_column='original_smiles', 
            property='pIC50', 
            feature_keys=['Bottleneck', 'rdkit', 'prolif', 'AffGraph'],
            protein_target='xyz', 
            pdb_folder='data/xyz_pdbs',
            sdf_file='data/ligands.sdf',
            processed_data_folder='data/xyz_processed/'
            checkpnt_nm='abs_stratified_leakproof.ckpt', 
            json_dict_file_nm='xyz_pIC50_dict.json')      

    Args:
        data_file: the csv file containing the data 
        df: the pandas dataframe containing the data, Optional, set to None if data needs to be loaded from the csv file in data_file
        smiles_column: the column name of the provided pandas dataframes containing the smiles
        property: the column name of the provided pandas dataframes containing the values to be modelled
        use_log10: whether to apply log10 transformation to regression properties, only use for positive values for the given property
        use_logit: whether to apply logit transformation to regression properties
        percentages: whether to divide the target by 100 in case the regression property is measured in percentages
        check_rdkit_desc: boolean to remove non valid smiles for rdkit descriptors, default true, Optional
        feature_keys: list of strings, each string is a feature key, which defines the feature_generator to be added to the list of features generators
        pdb_folder: contains a directory asd string to the folder containing the pdb files 
        sdf_file: contains the path to the sdf file containing the 3d ligand structures
        processed_data_folder: contains a directory where to store the processed data
        target: the name of the target
        property_key: the key used containing the value to model in the sdf_file
        chkpt_nm: the name of the checkpoint model to be loaded, can be abs_leakproof.ckpt, abs_lk_bn_val.ckpt or abs_stratified_leakproof.ckpt
        interactions: list of interactions to be modeled by prolif, can be any of Hydrophobic, HBDonor, HBAcceptor, PiStacking, Anionic, Cationic, CationPi, PiCation, VdWContact. Optional
        validation_strategy: defines the strategy to be used for creating the validation data set; options: 'stratified', 'leave_group_out', or 'mixed'
        use_available_data_split: defines wether to use the already available split if available in the data under the column data_split, default true
        clustering_method: method used for clustering similar compounds; options include 'Bottleneck', 'Butina', 'Scaffold'
        test_size: the proportion of data to use for validation
        random_state: random seed for reproducibility
        blender_properties: list of additionaly properties from the training data set to be used as features in the blender, Optional
        computational_load: indicates allowed execution time - 'cheap', 'intermediate', or 'expensive'
        cross_val_fold: number of folds for cross-validation
        model_nm: the name of model to save, default is 'automol.pt', must include .pt at the end
        pdf_title: The title of the pdf report of the model training
        pdf_authors: The authors of the report
        pdf_mails: The emails of the authors of the report
        pdf_summary: the summary of the model training
        pdf_file: the name of the file where to save the pdf report, must include .pdf at the end
        json_dict_file_nm: filename to save the dictionary, must include .json at the end

    Returns:
    The file where the dictionary is saved and the different keys of this dictionary. If the training is not succesfull, this will provide the error.  
   """
    #load the data for training
    if df is None:                                                                                  
        data_dict = load_data_for_training(data_file=data_file)                                  
        df = data_dict['data'] 

    # Prepare data for modeling 
    try:                                                                                     
        prepared_data = prepare_data_for_modeling(                                                                       
            df_given=df,                                                                                                 
            smiles_column=smiles_column,                                                                                        
            property=property,                                                                                         
            task='Regression',
            use_log10=use_log10, 
            use_logit=use_logit,
            percentages=percentages,
            check_rdkit_desc=check_rdkit_desc                                                                                         
        )
    except RuntimeError as e:
        return f'RunTimeError preparing data: {str(e)}'
        #raise McpError(
        #            ErrorData(
        #                    code=INTERNAL_ERROR,
        #                    message=f'Runtimeerror preparing data: {str(e)}'
        #                )
        #            ) 
    
    res = automol_model(
                        prepared_data=prepared_data,
                        data_file=data_file,
                        feature_keys=feature_keys,
                        pdb_folder=pdb_folder,
                        sdf_file=sdf_file,
                        processed_data_folder=processed_data_folder,
                        protein_target=protein_target,
                        property_key=property_key,
                        gradformer_chkpt_nm=gradformer_chkpt_nm,
                        interactions=interactions, 
                        validation_strategy=validation_strategy, 
                        use_available_data_split=use_available_data_split, 
                        clustering_method=clustering_method, 
                        test_size=test_size, 
                        random_state=random_state, 
                        blender_properties=blender_properties, 
                        computational_load=computational_load, 
                        cross_val_fold=cross_val_fold, 
                        model_nm=model_nm, 
                        pdf_title=pdf_title, 
                        pdf_authors=pdf_authors, 
                        pdf_mails=pdf_mails, 
                        pdf_summary=pdf_summary,
                        pdf_file=pdf_file,
                        json_dict_file_nm=json_dict_file_nm)   
    return res   


@mcp.tool()
def get_server_status() -> Dict[str, Any]:
    """
    Get current MCP server status, health information, and operational metrics.
    This tool follows MCP specification for server monitoring and diagnostics.
    
    Returns:
        Dictionary containing comprehensive server status and system information.
    """
    import psutil
    import platform
    from datetime import datetime
    
    try:
        # System information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status_info = {
            "server_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "server_info": SERVER_INFO,
            "available_tools": [
                "automol_regression_model",
                "automol_classification_model", 
                "list_tools",
                "get_server_status"
            ],
            "available_resources": [
                "molagent://docs/overview",
                "molagent://docs/features",
                "molagent://docs/examples",
                "molagent://docs/api",
                "molagent://docs/troubleshooting",
                "molagent://docs/agentic"
            ],
            "available_prompts": [
                "molagent"
            ],
            "system_information": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "hostname": platform.node()
            },
            "resource_usage": {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 2)
                },
                "cpu_percent": psutil.cpu_percent(interval=1)
            },
            "health_checks": {
                "memory_status": "good" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical",
                "disk_status": "good" if disk.used/disk.total < 0.8 else "warning" if disk.used/disk.total < 0.9 else "critical",
                "overall_health": "healthy"
            }
        }
        
        return status_info
        
    except Exception as e:
        return {
            "server_status": "operational_with_warnings",
            "error": f"Could not gather complete system information: {str(e)}",
            "basic_info": SERVER_INFO,
            "tools_available": True,
        }

if __name__ == "__main__":
    mcp.run(transport="sse", port=8001)
    #uvicorn.run(mcp, host="localhost", port=8001)
