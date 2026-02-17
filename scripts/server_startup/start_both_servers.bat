@echo off
REM MolAgent - Start Both Servers Script for Windows
REM This script starts both data and model servers in separate windows

echo Starting MolAgent Servers...
echo =============================

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment '.venv' not found!
    echo Please run the installation script first:
    echo   install.bat
    echo.
    pause
    exit /b 1
)

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

REM Check if startup scripts exist
if not exist "%SCRIPT_DIR%start_data_server.bat" (
    echo [ERROR] start_data_server.bat not found in %SCRIPT_DIR%!
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%start_model_server.bat" (
    echo [ERROR] start_model_server.bat not found in %SCRIPT_DIR%!
    pause
    exit /b 1
)

echo [INFO] Starting data server in new window...
start "MolAgent Data Server" cmd /k "%SCRIPT_DIR%start_data_server.bat"

echo [INFO] Waiting 3 seconds before starting model server...
timeout /t 3 /nobreak >nul

echo [INFO] Starting model server in new window...
start "MolAgent Model Server" cmd /k "%SCRIPT_DIR%start_model_server.bat"

echo.
echo [SUCCESS] Both servers are starting in separate windows.
echo.
echo Data Server: http://localhost:8000
echo Model Server: http://localhost:8001
echo.
echo For Claude Desktop integration, run:
echo   claude mcp add --transport sse automoldata https://localhost:8000/sse
echo   claude mcp add --transport sse automolmodelling https://localhost:8001/sse
echo.
echo Press any key to close this launcher window...
pause >nul