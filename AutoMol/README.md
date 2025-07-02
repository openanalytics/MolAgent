<img src="logo.png" width="100" height="100" align="right"> 

# AutoMol

## Install package

For automated pdf generation wkhtmltopdf is used. On linux install with
```{bash}
sudo apt-get install wkhtmltopdf
```
We recommend using an uv environment for this package.
```{bash}
pip install uv
uv venv automol_env --python 3.12
source automol_env/bin/activate
uv pip install automol_resources/
uv pip install automol/
uv pip install molfeat
uv pip install streamlit
uv pip install PyTDC
uv pip install torch_geometric prolif lightning
uv pip install rdkit==2024.3.5
uv pip install jupyter jupyterlab
uv run --with jupyter jupyter lab
```

Additionally, you can use the provided docker image, still requires the installation of AutoMol. 

## AutoMol

AutoMol is the python package used by MolAgent to built generic models for early-stage drug discovery.

### Concept

The idea of the AutoMoL package is to enable Machine Learning for non-experts and their project specific properties. This was made possible by two core concepts: 1) use of highly-informative features and 2) the combination of multiple shallow learners. The overall concept is detailed in the Figure 1 below. The pipeline only requires SMILES as input and a provided property target. These SMILES are first standardized.  Next, features are generated using these standardized smiles. The generated features are optionally given to feature selection or dimensionality reduction methods before training several base estimators. The predictions of these base estimators are then provided as input to train a final estimator or blender. The predictions of this final estimator is the final output. The AutoMoL pipeline can be used for regression or classification tasks.

<img src="Tutorials/hierarchy.png" width="700" height="250">

<sup> Figure 1. The concept of AutoMoL. Starting from the smiles, 
             features are generated and these features are used to
             train a combination of several shallow learners.</sup>

### Tutorials

Example Notebooks can be found in the folder Tutorials. In the tutorials we use data from [Therapeutic Data commons](https://tdcommons.ai/). A short summary for each notebook is given in the table below. 
| Notebook(s) | Summary |
| :-------------- | :---------------------------------------------------------- |
| Classifier, Regressor and RegressionClassifier| The most basic notebooks for regression and classification. These notebooks include examples of target transformation, use of sample weights, 3 predefined computational load settings, data splitting, visualizations and pdf generation. |
| Intermediate_Classifier, Intermediate_Regressor and Intermediate_RegressorClassifier | The intermediate notebooks for regression and classification. These notebooks include examples of functionality of the basic notebooks and adding feature generators, defining your own method hierarchy and dimensionality reduction.|
| Expert_Classifier, Expert_Regressor and Expert_RegressorClassifier | The expert notebooks for regression and classification. These notebooks include examples of functionality of the intermediate notebooks and how to add your own regressor/classifier, define/set your own hyperparameters and define your own clustering algorithm.|
| 3DRegressor | A notebook detailing the use of 3D feature generators such as prolif. |
| RelativeRegressor | A notebook detailing the use of relative ligand modelling. |
| BlenderFeatures |  A notebook detailing the use of feeding some features directly to the blender circumventing the base estimators.|
| Clustering_visualization | A notebook detailing visualizations of clustering results. |
| Data_cleaning | A notebook detailing limited data cleaning. |
| Manipulating_models| A notebook detailing how to manipulate trained automol models, merging models or deleting trained targets.|
| Molfeat_testing | A notebook detailing the available featuregenerators from molfeat. |
| MultiTarget_Classifier and MultiTarget_Regressor| A notebook detailing how to train a model for multiple targets at once. Instead of separate hierarchies for each target. |
| Plotly_pdf_generation| A notebook showing how to add your own plotly figure to the generated pdf. |

#### Tasks

We designed automol for three tasks: Classification, Regression and RegressionClassifier. This last term is defined for binary classification, where basicly model the classification problem as a regression problem with the target value being either 0 or 1. The output is then clipped to the interval [0,1] and used as the probability for the compound being classified as class 1.

#### 2D features

The 2D feature generation for chemical compounds is simply SMILES based. The default generators take as input a list of SMILES (strings) and return a data matrix containing the features. Automol has wrappers for the [molfeat](https://molfeat.datamol.io/) feature generators, as detailed in the Molfeat_testing notebook.  

#### 3D features
The automol has some structure aware features, such as prolif, see the 3Dregressor notebook in the folder Tutorials for more information. You can use these in automol if you provide 3d information in the form of an sdf file and pdb files. Al the different pdbs should be placed in the same folder. This folder should be provided. The sdf file contains all the structures of the compounds. There should be a property pdb referencing the name of the pdb file to be used. Next to the pdb name, the code also requires a property with the target value of the compound. For example, after unzipping <i>Data/manuscript_data.zip</i>,  <i>Data/manuscript_data/ABL/selected_dockings.sdf</i> contains the ligands and the pdbs are located in <i>Data/manuscript_data/ABL/pdbs</i>. 

### Python script with yaml file

A python script that reads the options from a yaml file is provided in the folder script. 

```{bash}
source automol_env/bin/activate
cd script/
uv run_automol.py --yaml_file automl_reg.yaml
```

### Streamlit App
Streamlit app for regression and classification can be found in the folder streamlit_app. Upload your csv file and start modelling. From the repository directory run:
```{bash}
source automol_env/bin/activate
cd streamlit_app/
uv run streamlit run automol_app.py
```


### Unittest

To execute unittests run the following command from the root directory of the repository:
```{bash}
source automol_env/bin/activate
cd automol/automol/
uv run -m unittest discover -cf
```

This does create model files in the execution directory to test saving and reloading models!

## Contacts

* **Authors**: Joris Tavernier, Marvin Steijaert
* **Contact**: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

&copy; All rights reserved, Open Analytics NV, 2021-2025.


