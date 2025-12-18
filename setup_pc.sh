#!/usr/bin/env bash

# Copyright (c) 2018-2025, Texas Instruments
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
# change the default here if needed - supported options: 11.2 11.1 11.0 10.1
TIDL_TOOLS_VERSION=${TIDL_TOOLS_VERSION-"11.2"}

# change the default here if needed - supported options: cpu gpu
# if you are invoking the script setup_runner_pc_gpu.sh, 
# then you would need to change this in that script and not here.
TIDL_TOOLS_TYPE=${TIDL_TOOLS_TYPE-"cpu"}

######################################################################
CURRENT_WORK_DIR=$(pwd)


#######################################################################
echo 'INFO: installing system dependencies...'

# Function to check if a package is installed using dpkg
is_package_installed() {
    local package_name="$1"
    if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "install ok installed"; then
        return 0  # Package is installed
    else
        return 1  # Package is not installed
    fi
}

# Function to install package only if not already installed
install_if_missing() {
    local package_name="$1"
    if is_package_installed "$package_name"; then
        echo "INFO: ✓ Package $package_name is already installed"
    else
        echo "INFO: installing $package_name..."
        sudo apt-get install -y "$package_name"
    fi
}

# Dependencies for cmake, onnx, pillow-simd, tidl-graph-visualization
# TBD: "graphviz-dev"
packages=("cmake" "libffi-dev" "libjpeg-dev" "zlib1g-dev" "protobuf-compiler" "graphviz")

for package in "${packages[@]}"; do
    install_if_missing "$package"
done


#######################################################################
pip3 install -e ./tools


######################################################################
# unsintall onnxruntime and install onnxruntime-tild along with tidl-tools
# pip3 uninstall -y onnxruntime onnxruntime-tidl


# tidlbenchmark-tools-download is a script that defined in and installed via tools/pyproject.toml
# tidlbenchmark-tools-download - this invokes: python3 tools/tidl_tools_package/download.py
echo "INFO: running tidlbenchmark-tools-download..."
TIDL_TOOLS_TYPE=${TIDL_TOOLS_TYPE} TIDL_TOOLS_VERSION=${TIDL_TOOLS_VERSION} tidlbenchmark-tools-download


######################################################################
pip3 install -e ./[pc] --verbose

# tidlbenchmark-tools-install is a script that defined in and installed via ./pyproject.toml
# download and install packages - this invokes: python3 edgeai_tidlrunner/download.py
# pip3 install --no-input onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com
# pip3 install --no-input osrt_model_tools @ git+https://github.com/TexasInstruments/edgeai-tidl-tools.git@11_00_08_00#subdirectory=osrt-model-tools
echo "INFO: running tidlbenchmark-tools-install..."
tidlbenchmark-tools-install


#######################################################################
# pillow-simd for faster resize
# there as issue with installing pillow-simd through requirements - force it here
# pip3 uninstall --yes pillow
# pip3 install --no-input -U --force-reinstall pillow-simd


######################################################################
if [ -d "${CURRENT_WORK_DIR}/../edgeai-tidl-tools" ]; then
  echo "WARNING: found local edgeai-tidl-tools at ${CURRENT_WORK_DIR}/../edgeai-tidl-tools/osrt-model-tools"
  echo "WARNING: using this local version instead of the one downloaded from the repository"
  pip3 uninstall -y osrt_model_tools
  cd ${CURRENT_WORK_DIR}/../edgeai-tidl-tools/osrt-model-tools
  python3 setup.py develop
  cd ${CURRENT_WORK_DIR}
fi


######################################################################
pip3 install -r requirements/nuscenes_requirements.txt


######################################################################
echo "INFO: setting pandaset Python package for 3D object detection"
echo "INFO: to use it, make sure the dataset has been downloaded as explained in ./docs/datsets.md"
./setup_pandaset.sh


#######################################################################
echo "INFO: to be able to use this package, datasets are required. some are automatically downloaded as the benchmark script is invoked."
echo "INFO: for the other datasets, the user may need to download manually, as explained in ./docs/setup_installation.md and ./docs/datasets.md"
echo "INFO: there are various options to restrict the datasets and models used - please see settings_base.yaml"
echo "INFO: completed installation."
