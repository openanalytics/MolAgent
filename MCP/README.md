<img src="../logo.png" width="100" height="100" align="right"> 

# MolAgent

The code for the tools is located in Tools and the mcp servers is mcp_server. The notebook **Lipophilicity_AstraZeneca.ipynb** uses gradio and smolagents for our Agentic AI framework. The notebook **MolAgent_multiagent.ipynb** provides several examples of how to use the framework. 

## Credentials
The notebook uses dotenv to store credentials
```{bash}
uv pip install python-dotenv
```
Create a file .env with the following content:

```{bash}
ANTHROPIC_API_KEY = xxxx
HF_TOKEN=xxxx
HF_HOME=hf_home/
TOKENIZERS_PARALLELISM=false
```
You can add any key you want in the .env file.

## Tools

We've created two MCP servers, one for data preparation and one for automol model training.

### Data Server Tools
| Tool |  Description|
| :--------------  | :---------------------------------------------------------- |
| retrieve_tdc_data | This tool retrieves the datasets defined by the given name from therapeutic data commons adme data and returns the location of the data file.  |
| retrieve_tdc_groups | Returns a list of the available problems or groups from the therapeutic data commons.  |
| retrieve_tdc_group_datasets | Returns a list of the possible dataset names from the therapeutic data commons for the given group or problem. |
| retrieve_3d_data | This tool reads the provided sdf file with 3d information and returns the location of a csv data file with the smiles and property value.|

### Model Server Tools

| Tool | Description|
| :-------------- | :---------------------------------------------------------- |
| automol_classification_model | This tool uses automol to train a classification model for chemical compounds for a particular property.|
| automol_regression_model | This tool uses automol to train a regression model for chemical compounds for a particular property. |


## Starting mcp servers locally
Using the automol_env environment, you can start the servers locally, by running the following commands in the terminal. We advise to run the servers from the notebook directory, since the mcp servers will save files only starting from the directory they are run from. 

Start data training server locally: 
```{bash}
source automol_env/bin/activate
cd MCP/
uv run mcp_server/automol_data_server.py
```
Start model training server locally:
```{bash}
source automol_env/bin/activate
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

