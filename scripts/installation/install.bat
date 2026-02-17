@echo off
REM MolAgent Installation Script for Windows
REM This script installs MolAgent using uv package manager

setlocal enabledelayedexpansion

REM Colors (limited support in Windows)
set "GREEN=[32m"
set "RED=[31m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "NC=[0m"

REM Default values
set "ENV_NAME=%~1"
if "%ENV_NAME%"=="" set "ENV_NAME=.venv"

set "PYTHON_VER=%~2"
if "%PYTHON_VER%"=="" set "PYTHON_VER=3.12"

echo %GREEN%[%date% %time%] Starting MolAgent installation...%NC%

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR] Python is required but not installed. Please install Python 3.8+ first.%NC%
    echo Visit: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo %BLUE%[INFO] Found Python version: %PYTHON_VERSION%%NC%

REM Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[INFO] Installing uv package manager...%NC%
    
    REM Try to install uv using pip
    python -m pip install uv
    
    REM Verify installation
    uv --version >nul 2>&1
    if errorlevel 1 (
        echo %RED%[ERROR] Failed to install uv. Please install manually:%NC%
        echo Visit: https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
) else (
    echo %GREEN%[INFO] Found uv package manager%NC%
)

REM Create virtual environment
echo %GREEN%[INFO] Creating virtual environment '%ENV_NAME%' with Python %PYTHON_VER%...%NC%
uv venv "%ENV_NAME%" --python "%PYTHON_VER%"
if errorlevel 1 (
    echo %RED%[ERROR] Failed to create virtual environment%NC%
    pause
    exit /b 1
)

REM Activate virtual environment
echo %GREEN%[INFO] Activating virtual environment...%NC%
call "%ENV_NAME%\Scripts\activate.bat"

REM Check for wkhtmltopdf
where wkhtmltopdf >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[WARNING] wkhtmltopdf not found in PATH%NC%
    echo %YELLOW%[WARNING] Please install wkhtmltopdf manually for PDF generation:%NC%
    echo %YELLOW%[WARNING] https://wkhtmltopdf.org/downloads.html%NC%
)

REM Install AutoMol submodule packages
echo %GREEN%[INFO] Installing AutoMol packages...%NC%
if exist "AutoMol\automol_resources" (
    uv pip install AutoMol/automol_resources/
) else (
    echo %YELLOW%[WARNING] AutoMol/automol_resources directory not found%NC%
    echo %YELLOW%[WARNING] Make sure to clone with submodules%NC%
)

if exist "AutoMol\automol" (
    uv pip install AutoMol/automol/
) else (
    echo %YELLOW%[WARNING] AutoMol/automol directory not found%NC%
    echo %YELLOW%[WARNING] Make sure to clone with submodules%NC%
)

REM Install essential dependencies first
echo %GREEN%[INFO] Installing essential dependencies...%NC%
uv pip install psutil>=5.9.0 smolagents>=1.19.0
if errorlevel 1 (
    echo %RED%[ERROR] Failed to install essential dependencies%NC%
    pause
    exit /b 1
)

REM Install MolAgent
echo %GREEN%[INFO] Installing MolAgent and dependencies...%NC%
uv pip install -e .
if errorlevel 1 (
    echo %RED%[ERROR] Failed to install MolAgent%NC%
    pause
    exit /b 1
)

REM Verify installation
echo %GREEN%[INFO] Verifying installation...%NC%
python -c "
import pandas as pd
import numpy as np
try:
    import rdkit
    import torch
    import fastmcp
    print('✓ Core dependencies installed successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"
if errorlevel 1 (
    echo %RED%[ERROR] Installation verification failed%NC%
    pause
    exit /b 1
)

REM Create .env template
echo %GREEN%[INFO] Creating .env template...%NC%
echo # MolAgent Environment Configuration > .env.template
echo # Copy this to .env and fill in your API keys >> .env.template
echo. >> .env.template
echo # Anthropic API Key for Claude integration >> .env.template
echo ANTHROPIC_API_KEY=your_anthropic_api_key_here >> .env.template
echo. >> .env.template
echo # Hugging Face Token for model downloads >> .env.template
echo HF_TOKEN=your_huggingface_token_here >> .env.template
echo HF_HOME=hf_home/ >> .env.template
echo. >> .env.template
echo # Disable tokenizers parallelism to avoid warnings >> .env.template
echo TOKENIZERS_PARALLELISM=false >> .env.template
echo. >> .env.template
echo # Optional: OpenAI API Key >> .env.template
echo OPENAI_API_KEY=your_openai_api_key_here >> .env.template
echo. >> .env.template
echo # Server Configuration >> .env.template
echo DATA_SERVER_PORT=8000 >> .env.template
echo MODEL_SERVER_PORT=8001 >> .env.template

REM Create startup scripts
echo %GREEN%[INFO] Creating startup scripts...%NC%

echo @echo off > start_data_server.bat
echo REM Start MolAgent Data Server >> start_data_server.bat
echo call %ENV_NAME%\Scripts\activate.bat >> start_data_server.bat
echo cd MCP >> start_data_server.bat
echo python mcp_server\automol_data_server.py >> start_data_server.bat
echo pause >> start_data_server.bat

echo @echo off > start_model_server.bat
echo REM Start MolAgent Model Server >> start_model_server.bat
echo call %ENV_NAME%\Scripts\activate.bat >> start_model_server.bat
echo cd MCP >> start_model_server.bat
echo python mcp_server\automol_model_server.py >> start_model_server.bat
echo pause >> start_model_server.bat

REM Success message
echo.
echo %GREEN%[SUCCESS] Installation completed successfully!%NC%
echo.
echo %BLUE%Next steps:%NC%
echo %BLUE%1. Copy .env.template to .env and add your API keys%NC%
echo %BLUE%2. Start the servers:%NC%
echo %BLUE%   - Both servers: start_both_servers.bat (recommended)%NC%
echo %BLUE%   - Data server only: start_data_server.bat%NC%
echo %BLUE%   - Model server only: start_model_server.bat%NC%
echo %BLUE%3. Or activate environment manually:%NC%
echo %BLUE%   %ENV_NAME%\Scripts\activate.bat%NC%
echo.
echo %BLUE%For Claude Desktop integration:%NC%
echo %BLUE%   claude mcp add --transport sse automoldata https://localhost:8000/sse%NC%
echo %BLUE%   claude mcp add --transport sse automolmodelling https://localhost:8001/sse%NC%
echo.
pause