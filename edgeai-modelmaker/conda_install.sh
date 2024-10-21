#!/usr/bin/env bash

#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

HOME_DIR=${HOME}
APPS_PATH=${HOME_DIR}/apps
DOWNLOADS_PATH=${HOME_DIR}/Downloads
CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
CONDA_URL=https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}
CONDA_PATH=${APPS_PATH}/conda
CONDA_BIN=${CONDA_PATH}/bin

echo "Installing MiniConda..." && \
  mkdir -p ${DOWNLOADS_PATH} && \
  if [ ! -f ${CONDA_INSTALLER} ]; then wget -q --show-progress --progress=bar:force:noscroll ${CONDA_URL} -O ${DOWNLOADS_PATH}/${CONDA_INSTALLER}; fi && \
  chmod +x ${DOWNLOADS_PATH}/${CONDA_INSTALLER} && \
  ${DOWNLOADS_PATH}/${CONDA_INSTALLER} -b -p ${CONDA_PATH}

echo "Adding conda init script to .bashrc, so that condabin is in PATH"
echo ". ${CONDA_PATH}/etc/profile.d/conda.sh" >> ${HOME_DIR}/.bashrc

echo "MiniConda has been installed."
