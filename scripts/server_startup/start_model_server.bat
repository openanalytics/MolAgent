@echo off
REM MolAgent Model Server Startup Script for Windows
REM This script starts the AutoMol model server (port 8001)

echo Starting MolAgent Model Server...
echo =================================

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment '.venv' not found!
    echo Please run the installation script first:
    echo   install.bat
    echo.
    pause
    exit /b 1
)

REM Check if MCP directory exists
if not exist "MCP" (
    echo [ERROR] MCP directory not found!
    echo Please ensure you're running this script from the MolAgent root directory.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo Please copy .env.template to .env and configure your API keys.
    echo The server will start but some features may not work properly.
    echo.
    timeout /t 3 /nobreak >nul
)

REM Check if port 8001 is already in use
echo [INFO] Checking port availability...
netstat -ano | findstr :8001 >nul
if not errorlevel 1 (
    echo [WARNING] Port 8001 is already in use!
    echo.
    echo Found processes using port 8001:
    netstat -ano | findstr :8001
    echo.
    echo Do you want to:
    echo [1] Kill existing process and start server
    echo [2] Exit and let you handle it manually
    echo.
    set /p choice="Enter your choice (1 or 2): "
    
    if "!choice!"=="1" (
        echo [INFO] Attempting to kill existing process on port 8001...
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do (
            taskkill /PID %%a /F >nul 2>&1
        )
        timeout /t 2 /nobreak >nul
        
        REM Check again if port is free
        netstat -ano | findstr :8001 >nul
        if not errorlevel 1 (
            echo [ERROR] Could not free port 8001. Please kill the process manually.
            echo.
            pause
            exit /b 1
        ) else (
            echo [SUCCESS] Port 8001 is now available.
        )
    ) else (
        echo [INFO] Exiting. Please stop the existing process and try again.
        pause
        exit /b 1
    )
)

REM Change to MCP directory
echo [INFO] Starting model server on port 8001...
cd MCP

REM Start the server
python mcp_server\automol_model_server.py

REM If we get here, the server has stopped
echo.
echo [INFO] Model server has stopped.
echo Press any key to close this window...
pause >nul