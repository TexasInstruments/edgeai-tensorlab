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

DOCKER_IMAGE_BASE="ubuntu:22.04"
DOCKER_IMAGE_NAME="edgeai-modelmaker-ubuntu:22.04-2025-05-05"

#################################################################################
USE_INTERNAL_REPO="1"

# docker and git repo locations - internal or external build
if [[ ${USE_INTERNAL_REPO} == "1" ]]; then
    DOCKER_REPO_LOCATION="artifactory.itg.ti.com/docker-public/library/"
    # SOURCE_LOCATION="ssh://git@bitbucket.itg.ti.com/edgeai-algo/"    
else
    DOCKER_REPO_LOCATION=""
    # SOURCE_LOCATION="https://github.com/TexasInstruments/"    
fi

# print
echo "DOCKER_REPO_LOCATION=${DOCKER_REPO_LOCATION}"
# echo "SOURCE_LOCATION=${SOURCE_LOCATION}"

# initialize http_proxy and https_proxy if they are not defined
http_proxy=${http_proxy:-""}
https_proxy=${https_proxy:-""}
no_proxy=${no_proxy:-""}

#################################################################################
CUR_DIR=${PWD}
PARENT_DIR=$(realpath ${CUR_DIR}/..)

#################################################################################
# Build docker image
DATE_TIME=$(date +'%Y%m%d-%H%M%S')
echo "building docker image at ${DATE_TIME} ..."

docker build \
    -f ${CUR_DIR}/docker/Dockerfile \
    -t ${DOCKER_IMAGE_NAME} \
    --build-arg DOCKER_IMAGE_BASE=${DOCKER_IMAGE_BASE} \
    --build-arg DOCKER_REPO_LOCATION=${DOCKER_REPO_LOCATION} \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy} \
    ${PARENT_DIR}
