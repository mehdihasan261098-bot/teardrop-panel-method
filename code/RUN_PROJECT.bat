@echo off
REM ===================================================================
REM  Panel Methods Project - Teardrop - ID 210011253
REM  Double-click this file to run the project.
REM ===================================================================
cd /d "%~dp0"

echo ===================================================================
echo  PANEL METHODS PROJECT - TEARDROP - ID 210011253
echo ===================================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found on this computer.
    echo Install Python from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during the installation.
    echo.
    pause
    exit /b 1
)

echo Checking that numpy and matplotlib are installed ...
python -c "import numpy, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Installing the missing packages ...
    python -m pip install --quiet numpy matplotlib
)

echo.
echo Running the project. This takes about half a minute.
echo.
python "teardrop_panel_method.py"

echo.
echo ===================================================================
echo  Finished. The figures are saved in the "outputs" folder.
echo ===================================================================
pause
