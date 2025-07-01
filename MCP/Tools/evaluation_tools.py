import pandas as pd
from typing import Any, Dict, List, Optional, TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

from smolagents import tool 

@tool 
def guide_prompt() -> str:
    """
    prompt to guide the managed agent
    """
    return f"""You are a professional data analytics manager overseeing a specialized team consisting of a Data Retriever and a Model Trainer. Your role is to coordinate their efforts and provide comprehensive analysis of results.
    
    WORKFLOW COORDINATION:
    1. ANALYSIS REQUEST ASSESSMENT
    - Determine if the request requires training a new model or using an existing model
    - Clearly identify data sources, target variables, and analysis objectives from the user prompt

    2. DATA PREPARATION PHASE (with Data Retriever)
    - Forward relevant user requirements to the Data Retriever agent
    - Request specific data formats and preprocessing steps based on analysis needs
    - Verify data quality, completeness, and format before proceeding
    - Maintain clear documentation of data provenance and transformations

    3. MODEL DEVELOPMENT PHASE (with Model Trainer)
    - Only when training a new model:
        * Provide user provided files, such as sdf files, csv files, pdb folders or data processing folders
        * Forward preprocessed data files and specific requirements to the Model Trainer
        * Specify evaluation metrics and validation approaches appropriate for the analysis goal
        * Request detailed model performance metrics and feature importance information
    - When using existing models:
        * Request specific inference results based on the provided data
        * Do NOT initiate new model training workflows

    4. RESULTS ANALYSIS AND PRESENTATION:
    - Provide comprehensive evaluation including:
        * Model performance metrics with contextual interpretation (not just raw numbers)
        * Feature importance analysis with business/domain implications
        * Comparative analysis if multiple approaches were tested
        * Potential limitations and recommended next steps
    - Generate professional visualizations:
        * For regression models: Scatter plots of actual vs. predicted values with regression line, residual plots, and feature importance charts
        * For classification models: Confusion matrices, ROC curves, precision-recall curves, and class distribution visualizations
        * Include relevant metrics in visualization titles or annotations (R², RMSE, F1, etc.)
        * Ensure all visualizations have proper labels, legends, and color schemes
    - Format results in a clear, professional structure with sections for:
        * Executive Summary (1-2 paragraphs)
        * Data Overview
        * Model Performance
        * Key Insights
        * Visualizations (with clear references to saved file names)
        * Recommendations

    COMMUNICATION GUIDELINES:
    - Maintain clear documentation of each step in the workflow
    - Use precise technical language appropriate for data science professionals
    - Provide context for technical metrics to make them accessible
    - Ensure all file references use consistent naming conventions
    - Clearly separate factual results from interpretative analysis

    Do not open json files yourself, this will fail. Use the provided tool to read json files. 
"""


@tool
def wait_for_llm_rate() -> str:
    """
    This tool let the llm breath. Call this tool before executing code
    """
    import time
    time.sleep(30)
    return 'succes'

@tool
def write_to_file(file_name:str='', content:str='') -> str:
    """
    This tool writes a gives content to a file. Do not do this yourself, use this tool instead.

    Args:
        file_name: the file name to write to
        content: the content to write to the file
    """
    import os 
    def sanitize_path(path):
        return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")
    file_name=sanitize_path(file_name)
    try:
        with open(file_name, 'w') as f: 
            f.write(content) 
    except Exception as e:
        return f'failed to write to {file_name}, does this file exists?, error: {str(e)} '
    return f'succesfully wrote given content to file {file_name}'

@tool
def read_json_file(model_performance_metrics_file:str=''  ) -> Dict[str,Any]:
    """
    This tool reads json files. Do not do this yourself.  


    Args:
        model_performance_metrics_file: the json file  
    """

    try: 
        import json
        with open(model_performance_metrics_file, 'r') as fp:
            data_dict = json.load(fp)
    except Exception as e:
        return f'failed to load json file {model_performance_metrics_file}, does this file exists?, error: {str(e)} '
    return data_dict

@tool
def load_automol_model(model_save_path:str=''  ) -> Dict[str,Any]:
    """
    This tool loads and then returns a a dictionary with the loaded automol model under the key model. 

    Args:
        model_save_path: the pt file containing the automol model
    """
    
    from automol.stacking import load_model
    stacked_model= load_model( model_save_path,use_gpu=False) 

    return {'model': stacked_model}

@tool
def automol_predict(model_dict:Dict[str,Any]={}, smiles_list:List[str]=None, blender_properties_dict:Optional[Dict[str,Any]]={}, transform_data:bool=False) -> Dict[str,Any]:
    """
    This tool uses the provided automol model and predicts properties for the given list of smiles strings. If the model is trained with blender properties the values of these properties for the given smiles must be given too.

    Args:
        model_dict: a dictionary with the automol model under the key model 
        smiles_list: a list of smiles strings for which the model make predictions
        blender_properties_dict: a dictionary containing the values of the features provided directly to the blender using their the feature or property names as keys. For each property a list of values corresponding to the value of given smiles is proved. 
        transform_data: return original predictions instead of transformed ones

    Returns:
        a dictionary with the predicted values 
    """
    assert 'model' in model_dict, f'Model not available under the key model in model_dict'
    return model_dict['model'].predict(props=None, smiles=smiles_list,blender_properties_dict=blender_properties_dict, compute_SD=True, convert_log10=transform_data)

