#!/bin/bash
set -eu

# change to script's directory (for requirements.txt)
PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$PARENT_PATH"

# check if conda is already installed
if command -v conda &> /dev/null ; then
    INSTALL_MINICONDA=0
else 
    INSTALL_MINICONDA=1
fi

PYTHON_VERSION=${1:-"3.9"}

if [[ $(uname) == "Linux" ]]; then
    MINICONDA_OS="Linux"
    MINICONDA_ARCH=$(uname -m)
elif [[ $(uname) == "Darwin" ]]; then
    MINICONDA_OS="MacOSX"
    MINICONDA_ARCH=$(uname -m)
    PYTHON_VERSION="3.9"
    echo "MacOSX only supported with python 3.9"
else
    echo "Unsupported operating system: $(uname)"
    exit 1
fi

if [ $INSTALL_MINICONDA -eq 1 ] ; then
    # Specify the Miniconda installation directory
    MINICONDA_DIR="$HOME/miniconda"

    MINICONDA_FILE="Miniconda3-latest-${MINICONDA_OS}-${MINICONDA_ARCH}.sh"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_FILE"

    echo "Downloading Miniconda installation script..."
    curl -O $MINICONDA_URL

    echo "Installing miniconda to $HOME/miniconda"
    bash $MINICONDA_FILE -b -p $MINICONDA_DIR

    # Add the Miniconda bin directory to PATH
    export PATH="$HOME/miniconda/bin:$PATH"

    # Clean up the installation files
    rm $MINICONDA_FILE
else 
    echo "conda already installed"
fi

if ! conda env list | grep -q "^degirum\s"; then
    # Create a new environment called "degirum" with the specified Python version.
    echo "Creating the degirum environment"
    conda create --yes -n degirum python=$PYTHON_VERSION pip

    # Install python requirements in degirum environment
    eval "$(conda shell.bash hook)"
    conda activate degirum
    pip install -r requirements.txt
    python -m ipykernel install --user --name degirum --display-name "Python (degirum)"
    conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
    
    echo "The degirum conda environment has been installed!"
else 
    echo "The degirum conda environment already exists"
fi 

if [ $INSTALL_MINICONDA -eq 1 ] ; then
    conda init bash
fi

echo "Activate with 'conda activate degirum'"
echo "Launch jupyterlab server by running 'jupyter lab' from the PySDKExamples directory"

# Launch a new bash with activated environment 
bash --rcfile <(echo '. ~/.bashrc; conda activate degirum')
