@echo off
REM MolAgent Master Script Launcher
REM Interactive menu for all MolAgent operations

setlocal enabledelayedexpansion

:MAIN_MENU
cls
echo.
echo  ╔══════════════════════════════════════╗
echo  ║        MolAgent Script Launcher      ║
echo  ╠══════════════════════════════════════╣
echo  ║                                      ║
echo  ║  1. Install MolAgent                 ║
echo  ║  2. Start Both Servers               ║
echo  ║  3. Start Data Server Only           ║
echo  ║  4. Start Model Server Only          ║
echo  ║  5. View Installation Status         ║
echo  ║  6. Open Scripts Directory           ║
echo  ║  7. Help & Documentation             ║
echo  ║  8. Exit                             ║
echo  ║                                      ║
echo  ╚══════════════════════════════════════╝
echo.

set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto START_BOTH
if "%choice%"=="3" goto START_DATA
if "%choice%"=="4" goto START_MODEL
if "%choice%"=="5" goto STATUS
if "%choice%"=="6" goto OPEN_SCRIPTS
if "%choice%"=="7" goto HELP
if "%choice%"=="8" goto EXIT

echo Invalid choice. Please select 1-8.
pause
goto MAIN_MENU

:INSTALL
cls
echo Starting MolAgent Installation...
echo ================================
echo.
call installation\install.bat
echo.
echo Installation completed. Press any key to return to menu...
pause >nul
goto MAIN_MENU

:START_BOTH
cls
echo Starting Both MolAgent Servers...
echo =================================
echo.
call server_startup\start_both_servers.bat
echo.
echo Servers started. Press any key to return to menu...
pause >nul
goto MAIN_MENU

:START_DATA
cls
echo Starting MolAgent Data Server...
echo ================================
echo.
call server_startup\start_data_server.bat
echo.
echo Data server started. Press any key to return to menu...
pause >nul
goto MAIN_MENU

:START_MODEL
cls
echo Starting MolAgent Model Server...
echo =================================
echo.
call server_startup\start_model_server.bat
echo.
echo Model server started. Press any key to return to menu...
pause >nul
goto MAIN_MENU

:STATUS
cls
echo MolAgent Installation Status
echo ============================
echo.

REM Check virtual environment
if exist "molagent_env\Scripts\activate.bat" (
    echo [✓] Virtual environment: Found
) else (
    echo [✗] Virtual environment: Not found
)

REM Check if servers are running
netstat -ano | findstr :8000 >nul 2>&1
if not errorlevel 1 (
    echo [✓] Data Server (port 8000): Running
) else (
    echo [✗] Data Server (port 8000): Not running
)

netstat -ano | findstr :8001 >nul 2>&1
if not errorlevel 1 (
    echo [✓] Model Server (port 8001): Running
) else (
    echo [✗] Model Server (port 8001): Not running
)

REM Check .env file
if exist ".env" (
    echo [✓] Configuration file (.env): Found
) else (
    echo [✗] Configuration file (.env): Not found
)

echo.
echo Press any key to return to menu...
pause >nul
goto MAIN_MENU

:OPEN_SCRIPTS
cls
echo Opening Scripts Directory...
echo ===========================
echo.
explorer .
echo Scripts directory opened in Explorer.
echo Press any key to return to menu...
pause >nul
goto MAIN_MENU

:HELP
cls
echo MolAgent Help & Documentation
echo =============================
echo.
echo Available Documentation:
echo - README.md (in scripts folder) - Complete script documentation
echo - INSTALL.md (in root) - Installation guide
echo - CLAUDE.md (in root) - Development commands
echo - SERVER_STARTUP_GUIDE.md (in root) - Server management
echo.
echo Quick Commands:
echo - Installation: scripts\installation\install.bat
echo - Start Servers: scripts\server_startup\start_both_servers.bat
echo - Data Server: scripts\server_startup\start_data_server.bat
echo - Model Server: scripts\server_startup\start_model_server.bat
echo.
echo For issues: Check logs and script output for error details
echo.
echo Press any key to return to menu...
pause >nul
goto MAIN_MENU

:EXIT
echo.
echo Thank you for using MolAgent!
echo.
exit /b 0