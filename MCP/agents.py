from Tools.tdc_tool import retrieve_tdc_data, retrieve_tdc_groups, retrieve_tdc_group_datasets 
from Tools.dataset_tools import retrieve_3d_data, data_answer, check_valid_smiles
from Tools.training_tools import load_data_for_training, train_automol_model, create_validation_split, get_feature_generators, add_affinity_graph_feature_generator, training_answer
from Tools.training_tools import prepare_data_for_modeling, evaluate_automol_model, add_prolif_feature_generator
from Tools.evaluation_tools import read_json_file, write_to_file, wait_for_llm_rate

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    ToolCollection
)

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "gradio",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "scipy.stats",
    'seaborn',
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "json",
    "torch",
    "datetime",
    "fractions",
    "rdkit",
    "rdkit.Chem",
    "rdkit.Chem.Descriptors",
    "stats",
    "csv",
    "posixpath",
    "matplotlib.style",
    "matplotlib.pyplot",
    "matplotlib.show",
    "matplotlib"
]

def get_data_agent(model=None,tools=None,max_steps=10):
    if tools is None:
        tools=[retrieve_tdc_data, retrieve_tdc_groups, retrieve_tdc_group_datasets, retrieve_3d_data]
    data_agent = CodeAgent(
        tools=[ *tools, data_answer, check_valid_smiles, write_to_file, wait_for_llm_rate],
        model=model,
        max_steps=max_steps,
        name="dataset_loader_agent",
        description="""    
        A team member that retrieves data for you. Provide him with as much information as possible related to the data. 
        You can provide him with:
        1. directory of a csv file
        2. name from dataset names from therapeutics data commons
        3. Provide a sdf file accompanied with a pdb file folder
        
        Your request should be a real sentence. You will receive the directory of the data file, column with the smiles and the target column. These are important for other team members, remember these and provide them to other team members as well as the sdf file and pdb folder if given in the user prompt.""",
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        provide_run_summary=False,
    )
    
    data_agent.prompt_templates["managed_agent"]["task"] = """
    You're a helpful agent named '{{name}}' specializing in cheminformatics and molecular data science.
    You have been submitted this task by your manager who needs prepared data for molecular property prediction.
    ---
    Task:
    {{task}}
    ---

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
    """
    
    return data_agent

def get_mcp_model_agent(model=None,tools=None,max_steps=10):
    if tools is None:
        return get_model_agent(model,max_steps)
    else:
        model_agent = CodeAgent(
            tools=[*tools, check_valid_smiles, read_json_file, write_to_file, wait_for_llm_rate],
            model=model,
            max_steps=max_steps,
            name="model_training_agent",
            description="""    
            A team member that prepares and splits the data, trains a model and evaluates a model for you using automol. Provide this team member All of the following information when known:
            1. the directory of the data file saved and the target column by the team member that retrieved the data for you,
            2. the smiles column, target column, length of the data and
            3. the original sdf file containing the ligands, the pdb folder with proteins and possibly the already processed data folder from the user prompt.
            
            Provide him with as much additional information as possible related to the data and the task. 
            
            Your request should be a real sentence. You will receive a directory where the performance metrics dictionary is saved as json file. This are important for further processing.""",
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            provide_run_summary=False,
        )
        model_agent.prompt_templates["managed_agent"]["task"] = """
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
        In all cases, provide your manager with the json file(s) including all the metrics of the model(s), this is very important! 
        """
        return model_agent


def get_model_agent(model=None,max_steps=10):
    
    model_agent = CodeAgent(
        tools=[load_data_for_training, check_valid_smiles, prepare_data_for_modeling, train_automol_model, create_validation_split,
               get_feature_generators, add_prolif_feature_generator, add_affinity_graph_feature_generator, evaluate_automol_model, training_answer,
                 check_valid_smiles, read_json_file, write_to_file],
        model=model,
        max_steps=max_steps,
        name="model_training_agent",
        description="""    
        A team member that prepares and splits the data, trains a model and evaluates a model for you using automol. Provide this team member All of the following information when known:
        1. the directory of the data file saved and the target column by the team member that retrieved the data for you,
        2. the smiles column, target column, length of the data and
        3. the original sdf file containing the ligands and the pdb folder with proteins from the user prompt.
        
        Provide him with as much additional information as possible related to the data and the task. 
        
        Your request should be a real sentence. You will receive a directory where the performance metrics dictionary is saved as json file. This are important for further processing.""",
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        provide_run_summary=False,
    )
    model_agent.prompt_templates["managed_agent"]["task"] = """
    You're a helpful agent named '{{name}}'.
    You have been submitted this task by your manager.
    ---
    Task:
    {{task}}
    ---
    You're a data scientist in charge of splitting the data and training an automol model. Proceed in the following order:
    
    1. Use the tool load_data_training to get load the data from the data preparation stage. 
    2. Verify the SMILES column by using the tool check_valid_smiles. If the user did not provide a column containing the SMILES,
    try out columns from the dataframe with the name smiles, original_smiles, structures, drug or columns containing these words. Do not try columns with ID in their name.         
    3.  Perform data analysis on the target. Determine the skewness of the target. For the remaining columns of available in the dataset, try to compute the correlation with
    the target. Compute the number of unique values of the target.
    4. Prepare the data using the prepare_data_for_modeling tool using the information from the user prompt. If the task is not specified, try regression first.
    If the values of the target are positive and the absolute value of the skewness is below 0.3 use log10 transform. If the task is classification, look for information about the
    classes from the user. If threshold values of class quantiles are given for classification, use these. If no class thresholds or quantiles are given and the user prompt says the target is categorical
    or the number of unique values of the target is less than 6, the target is categorical. If the task is classification and no other information is provided and the 
    target is not categorical, divide the target in two classes by using quantile 0.5 in the tool prepare_data_for_modeling.   
    5. Collect the different feature generators you want to test by using the tools get_feature_generators, add_prolif_feature_generator and add_affinity_graph_feature_generator. 
    Use the information from the user. If no information can be found try the default features generators:  Bottleneck, rdkit and fps_1024_2. You can use these as standalone 
    or combine them. If the user_prompt provides information on a sdf file and pdb folder, add prolif with all interactions as feature generator too. 
    6. Split the data in training and validation. Do this only once! There are two approaches:
        - If the user askes for a specific strategy or no validation is provided, use the tool create_validation_split. Use the provided information, if no details are given about the strategy used mixed and 
        if this fails use stratified. 
        - If no strategy is provided and the data set contains data_split, use use_available_data_split in the tool create_validation_split.
    7. Train a model using the tool train_automol_model, given the provided feature generator keys, try out different feature combinations. Use positively correlated columns to the target from the data set
    as blender_properties. Look for information in the user prompt regarding computational load, if not information is found use the cheapest. 
    8. Evaluate the model using the tool evaluate_automol_model. Collect the different metrics for all the different feature combinations or options.
    9. Apply the tool training_answer to save the performance metrics. Use this information in your final answer, provide your manager with the location of the file with model performance metrics.  
    
    You are helping your manager with a broader data science task, make sure to not provide a one-line answer, but give as much information as possible to give them a clear understanding of the answer.
    
    And even if your task resolution is not successful, please return as much context as possible, so that your manager can act upon this feedback. You can even ask for more information or clarification or missing information about files.
    
    At all costs provide your manager with the file name where the metrics are saved in the tool training_answer, this is critical! 
    
    Your final_answer WILL HAVE to contain these parts:
    ### 1. Task outcome (short version): 
    ### 2. Metrics file, as well as short descriptions of number of models trained and the used parameters:
    ### 3. Additional context (if relevant):
    
    Put all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be lost.
    """
    return model_agent