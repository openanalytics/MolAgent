<img src="../molagent.png" width="100" height="100" align="right"> 

# MolAgent

The code for the tools is located in Tools and the mcp servers is mcp_server. The notebook **Lipophilicity_AstraZeneca.ipynb** uses gradio and smolagents for our Agentic AI framework. The notebook **MolAgent_multiagent.ipynb** provides several examples of how to use the framework. Make sure that the MCP servers are running see [README](../README.md). 

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


