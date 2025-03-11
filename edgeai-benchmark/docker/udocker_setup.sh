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

UDOCKER_INSTALL_PATH=${1:-~/}
UDOCKER_DIR=${2:-~/"${UDOCKER_INSTALL_PATH}/udocker_home"}
CUR_DIR=$(pwd)

echo "UDOCKER_INSTALL_PATH=" ${UDOCKER_INSTALL_PATH}
echo "UDOCKER_DIR=" ${UDOCKER_DIR}

cd ${UDOCKER_INSTALL_PATH}
wget https://github.com/indigo-dc/udocker/releases/download/1.3.5/udocker-1.3.5.tar.gz
tar zxvf udocker-1.3.5.tar.gz
export PATH=${UDOCKER_INSTALL_PATH}/udocker:$PATH

curl -L https://github.com/jorge-lip/udocker-builds/raw/master/tarballs/udocker-englib-1.2.8.tar.gz > udocker-englib-1.2.8.tar.gz
export UDOCKER_TARBALL=udocker-englib-1.2.8.tar.gz
udocker install

cd ${CUR_DIR}
