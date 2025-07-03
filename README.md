
<img src="molagent.png" width="100" height="100" align="right"> 

# MolAgent

The advent of agentic AI systems is leading to significant transformations acrossscientific and technological domains. Computer-aided drug design (CADD)—a multifaceted process encompassing complex, interdependent tasks—stands to benefitprofoundly from these advancements. However, a challenge is empowering agentic systems to autonomously construct models for properties estimation that match the quality and reliability of those developed by human experts. As this is not currently straight forward, this capability represents a major bottleneck for fully realizing the potential of autonomous pipelines in drug discovery. We present here MolAgent, a system-agnostic agentic AI framework designed for high-fidelity modeling of molecular properties in early-stage drug discovery. MolAgent autonomously implements expert-level pipelines for both classification and regression, empowering agentic systems to efficiently construct and deploy models. With integrated automated feature engineering, robust model selection, advanced ensemble methodologies, and comprehensive validation frameworks, MolAgent ensures optimal accuracy and model robustness. The platform seamlessly accepts 2D and 3D structural data for ligands and receptors and harmonizes traditional molecular descriptors with advanced deep learning features extracted from pretrained 2D and 3D encoders. Ultimately the platform’s fully automated, end-to-end workflow is designed for seamless agentic execution. Adherence to the Model Context Protocol (MCP) guarantees interoperability with diverse agenticAI infrastructures, ensuring flexible integration into complex, future discovery pipelines.

MolAgent enables the following capabilities:
* Model Context Protocol (MCP) servers, facilitating seamless integration with agentic systems to support enhanced autonomous workflows. The MCP-ready design of MolAgent empowers agentic AI by enabling autonomous decision-making, automated iterative experimentation, and dynamic workflow management. This capability significantly enhances operational efficiency in complex environments, facilitating real-time adaptation and optimized modeling through intelligent agentic interactions.
* Integration of diverse molecular representation, from traditional chemical or
similar fingerprints to modern deep learning embeddings.
* Explicit incorporation of 3D structural information and protein-ligand
interactions.
* Specialized data splitting strategies that respect chemical series and activity cliffs.
* Automated model selection through nested cross-validation methods.
* Implementation of multiple model ensemble strategies tailored to molecular
property estimation tasks.
* Comprehensive validation procedures designed for chemical data.

## Install package

For automated pdf generation wkhtmltopdf is used. On linux install with
```{bash}
sudo apt-get install wkhtmltopdf
```
We recommend using an uv environment for this package. The MCP server uses AutoMol which is cloned from the repository. 
```{bash}
git clone https://github.com/openanalytics/AutoMol
pip install uv
uv venv molagent_env --python 3.12
source molagent_env/bin/activate
uv pip install AutoMol/automol_resources/
uv pip install AutoMol/automol/
uv pip install molfeat
uv pip install streamlit
uv pip install PyTDC
uv pip install torch_geometric prolif lightning
uv pip install rdkit==2024.3.5
uv pip install transformers smolagents[all] fastmcp
uv pip install jupyter jupyterlab
uv run --with jupyter jupyter lab
```

Additionally, you can use the provided docker image, still requires the installation of AutoMol. 

## MCP servers and Agentic framework

The MCP servers for used by the agentic AI systems are provided in the folder MCP. This folder additionaly contains the prompts for three agents: a data agent, modelling agent and a manager agent. A notebook containing the smolagents example using gradio is provided. 

## Tools

We've created two MCP servers, one for data preparation and one for automol model training.


### Model Server Tools

| Tool | Description|
| :-------------- | :---------------------------------------------------------- |
| automol_classification_model | This tool uses automol to train a classification model for chemical compounds for a particular property.|
| automol_regression_model | This tool uses automol to train a regression model for chemical compounds for a particular property. |

### Auxiliary mcp server to retrieve public database

| Tool |  Description|
| :--------------  | :---------------------------------------------------------- |
| retrieve_tdc_data | This tool retrieves the datasets defined by the given name from therapeutic data commons adme data and returns the location of the data file.  |
| retrieve_tdc_groups | Returns a list of the available problems or groups from the therapeutic data commons.  |
| retrieve_tdc_group_datasets | Returns a list of the possible dataset names from the therapeutic data commons for the given group or problem. |
| retrieve_3d_data | This tool reads the provided sdf file with 3d information and returns the location of a csv data file with the smiles and property value.|



## Starting mcp servers locally
Using the molagent_env environment, you can start the servers locally, by running the following commands in the terminal. We advise to run the servers from the notebook directory, since the mcp servers will save files only starting from the directory they are run from. 

Start data training server locally: 
```{bash}
source molagent_env/bin/activate
cd MCP/
uv run mcp_server/automol_data_server.py
```
Start model training server locally:
```{bash}
source molagent_env/bin/activate
cd MCP/
uv run mcp_server/automol_model_server.py
```
In the terminal of the model server, you can follow the progress of the model training. 

## Tool Inspector

You can start the MCP tool inspector by running:
```
npx @modelcontextprotocol/inspector
```
Make sure to copy the session token and set it as Proxy Session Token (under configuration) in the inspector GUI. Then set transport type as SSE with either 
```
http://localhost:8001/sse
```
or
```
http://localhost:8000/sse
``` 
as URL.

## Contacts

* **Authors**: Joris Tavernier, Marvin Steijaert
* **Contact**: joris.tavernier@openanalytics.eu, Marvin.Steijaert@openanalytics.eu

&copy; All rights reserved, Open Analytics NV, 2021-2025.
