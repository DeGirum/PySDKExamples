@echo off
setlocal enabledelayedexpansion

REM change to script's directory (for requirements.txt)
pushd "%~dp0"

REM check if conda is already installed
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "INSTALL_MINICONDA=0"
) else (
    set "INSTALL_MINICONDA=1"
)

set "PYTHON_VERSION=3.9"
IF NOT "%1"=="" (
    SET PYTHON_VERSION=%1
)

echo ",3.8,3.9,3.10,3.11," | findstr /C:",%PYTHON_VERSION%," >nul
if %errorlevel% neq 0 (
    echo "Unsupported Python version !PYTHON_VERSION!. supported versions are 3.8, 3.9, 3.10, 3.11"
    exit /b 1
)

set "MINICONDA_DIR=%USERPROFILE%\miniconda3"
if %INSTALL_MINICONDA% EQU 1 (
    echo Downloading miniconda installer.
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
    if !ERRORLEVEL! neq 0 (
        echo Failed to download miniconda installer
        exit /b !ERRORLEVEL!
    ) 
    
    echo Installing miniconda to %MINICONDA_DIR%
    call miniconda.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D=%MINICONDA_DIR%
    if !ERRORLEVEL! neq 0 (
        echo Failed to install miniconda 
        exit /b !ERRORLEVEL!
    ) 

    REM Add the Miniconda Scripts directory to PATH
    set "PATH=%MINICONDA_DIR%\Scripts;%PATH%"

    REM Clean up the installation files
    del "miniconda.exe"
) else (
    echo conda already installed
)

conda env list | findstr /B /C:"degirum " >nul
if %errorlevel% neq 0 (
    REM Create a new environment called "degirum" with the specified Python version.

    echo Creating the degirum environment
    call conda create --yes -n degirum python=%PYTHON_VERSION% pip
    if !ERRORLEVEL! neq 0 (
        echo Failed to create degirum environment
        exit /b !ERRORLEVEL!
    ) 

    REM Install python requirements in degirum environment
    call activate degirum
    if !ERRORLEVEL! neq 0 (
        echo Failed to activate degirum environment
        exit /b !ERRORLEVEL!
    ) 
    call pip install -r requirements.txt
    if !ERRORLEVEL! neq 0 (
        echo Failed to install requirements
        exit /b !ERRORLEVEL!
    ) 
    python -m ipykernel install --user --name degirum --display-name "Python (degirum)"
    if !ERRORLEVEL! neq 0 (
        echo Failed to add degirum kernel
        exit /b !ERRORLEVEL!
    ) 

    echo The degirum conda environment has been installed!
) else (
    echo The degirum conda environment already exists.
)


if %INSTALL_MINICONDA% EQU 1 (
    call conda init cmd.exe
)

echo Activate degirum conda environment with 'conda activate degirum'
echo Launch jupyterlab server by running 'jupyter lab' from the PySDKExamples directory
popd
pause
start cmd /k activate degirum