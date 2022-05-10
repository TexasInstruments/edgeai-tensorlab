#!/usr/bin/env bash

# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################
version_match=`python3 -c 'import sys;r=0 if sys.version_info >= (3,6) else 1;print(r)'`
if [ $version_match -ne 0 ]; then
echo 'python version must be >= 3.6'
exit 1
fi

######################################################################
echo "pycocotools need cython has to be installed from conda, if this is conda python"
conda install -y cython

######################################################################
# Dependencies for building pillow-simd
#echo 'Installing dependencies to build pillow-simd. If you dont have sudo access, comment the below line and replace pillow-simd with pillow in the requirements file'
#sudo apt-get -y install libjpeg-dev zlib1g-dev

echo 'Installing python packages...'
pip3 install --no-input -r ./requirements_pc.txt

######################################################################
#NOTE: THIS STEP INSTALLS THE EDITABLE LOCAL MODULE pytidl
echo 'Installing as a local module using setup.py'
python3 setup.py develop

######################################################################
# Installing dependencies
echo 'Cleaning up previous tidl_tools...'
rm -rf tidl_tools.tar.gz tidl_tools

echo 'Installing tidl_tools...'

# installers for 8.2 release
pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/dlr-1.10.0-py3-none-any.whl
pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_05/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/tidl_tools.tar.gz

# installers for 8.1 release
#pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/dlr-1.8.0-py3-none-any.whl
#pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
#pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_01_00_09-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
#pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
#wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_01_00_09-rc1/tidl_tools.tar.gz

# installers for 8.0 release
#pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/dlr-1.8.0-py3-none-any.whl
#pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tvm-0.8.dev0-cp36-cp36m-linux_x86_64.whl
#pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl
#pip3 install --no-input https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tflite_runtime-2.4.0-py3-none-any.whl
#wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08.00.00-rc1/tidl_tools.tar.gz

tar -xzf tidl_tools.tar.gz

######################################################################
export TIDL_TOOLS_PATH=$(pwd)/tidl_tools
echo "TIDL_TOOLS_PATH=${TIDL_TOOLS_PATH}"


######################################################################
cd $TIDL_TOOLS_PATH

echo "[gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu] Checking ..."
if [ ! -d gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu ]
then
    wget https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz --no-check-certificate
    tar xf gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz > /dev/null
    rm gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
fi
cd ..
echo "[gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu] Done"

######################################################################
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# make sure current directory is visible for python import
export PYTHONPATH=:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"

# needed for TVM compilation
export ARM64_GCC_PATH=$TIDL_TOOLS_PATH/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

echo 'Completed installation.'
