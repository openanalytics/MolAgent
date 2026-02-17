# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MolAgent is a multi-agent system for AI-driven molecular property prediction in early-stage drug discovery. It provides MCP (Model Context Protocol) servers that wrap the AutoMol ML framework, enabling agentic AI systems to autonomously train predictive models for molecular properties.

## Key Commands

### Installation
```bash
# Clone with submodules (required for AutoMol backend)
git clone --recurse-submodules https://github.com/openanalytics/MolAgent

# Install (creates .venv with Python 3.12)
./install.sh

# Manual installation
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install pytdc rdkit==2024.3.5
uv pip install AutoMol/automol_resources/ AutoMol/automol/
```

### Starting MCP Servers
```bash
# From repository root, in separate terminals:
source .venv/bin/activate && cd MCP && uv run mcp_server/automol_data_server.py     # Port 8000
source .venv/bin/activate && cd MCP && uv run mcp_server/automol_model_server.py    # Port 8001

# Or use startup scripts:
./scripts/server_startup/start_data_server.sh
./scripts/server_startup/start_model_server.sh
```

### Testing
```bash
# AutoMol unit tests
cd AutoMol/automol && uv run -m unittest discover -cf

# Plugin tests (from automol-tasks-manager/)
uv run pytest tests/
```

### Claude Code Plugin
```bash
# Use as Claude Code plugin with train-pipeline and predict skills
claude --plugin-dir ./automol-tasks-manager/
```

## Architecture

```
MolAgent/
├── AutoMol/                    # ML backend (git submodule)
│   ├── automol/                # Core AutoMol library
│   └── automol_resources/      # Precomputed molecular features
├── MCP/                        # MCP server layer
│   ├── mcp_server/
│   │   ├── automol_model_server.py  # Port 8001 - regression/classification training
│   │   └── automol_data_server.py   # Port 8000 - TDC data retrieval, 3D processing
│   ├── Tools/training_tools.py      # Training functions called by model server
│   ├── GradioMolAgent.py            # Gradio chatbot interface
│   └── agents.py                    # SmolAgents multi-agent orchestration
├── automol-tasks-manager/      # Claude Code plugin
│   ├── skills/train-pipeline/  # Complete SMILES-to-model workflow
│   ├── skills/predict/         # Inference with trained models
│   └── hooks/setup-automol-env.sh  # Exports AUTOMOL_ROOT, PLUGIN_ROOT
├── MolagentFiles/              # Pipeline outputs (run folders, model_registry.json)
└── Data/                       # Sample datasets
```

## MCP Server Tools

**Model Server (port 8001):**
- `automol_regression_model` - Train regression models for continuous properties
- `automol_classification_model` - Train classification models for categorical properties

**Data Server (port 8000):**
- `retrieve_tdc_data` - Download datasets from Therapeutic Data Commons
- `retrieve_3d_data` - Process SDF files with 3D structures and PDB files

## Pipeline State (automol-tasks-manager)

Pipeline runs create isolated folders: `MolagentFiles/{dataset}-{props}-{timestamp}/`

State uses `"outputs"` key for file paths:
```python
outputs = state.get("outputs", state.get("files", {}))
```

Key conventions:
- SMILES column standardized to `Stand_SMILES` after preparation
- Model files: `{property}_stackingregmodel.pt` or `{property}_stackingclfmodel.pt`
- Refitted models have `_refitted` suffix in filename
- Merged models combine per-property files to eliminate encoder duplication

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `AUTOMOL_ROOT` | Plugin root directory (set by SessionStart hook) |
| `AUTOMOL_VENV` | Virtual environment name (default: `.venv`) |

## Dependencies

Key packages: rdkit==2024.3.5, molfeat, prolif, scikit-learn, torch, fastmcp, smolagents, pytdc

Requires Python 3.8+ (3.12 recommended), uv package manager, and wkhtmltopdf for PDF generation.
