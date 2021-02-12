#!/bin/bash

######################################################################
version_match=`python -c 'import sys;r=0 if sys.version_info >= (3,6) else 1;print(r)'`
if [ $version_match -ne 0 ]; then
echo 'python version must be >= 3.6'
exit 1
fi

######################################################################
echo "pycocotools need cython has to be installed from conda, if this is conda python"
conda install -y cython

######################################################################
# Installing dependencies
echo 'Installing python packages...'
pip install -r ./requirements.txt

######################################################################
#NOTE: THIS STEP INSTALLS THE EDITABLE LOCAL MODULE pytidl
echo 'Installing as a local module using setup.py'
pip install -e ./





