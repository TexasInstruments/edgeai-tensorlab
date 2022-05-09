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

# determine if we are behind ti firewall
ping bitbucket.itg.ti.com -c 1 > /dev/null 2>&1
PING_CHECK="$?"
# internal or external build
if [ ${PING_CHECK} -eq "0" ]; then
    REPO_LOCATION=artifactory.itg.ti.com/docker-public/library/
    PROXY_LOCATION=http://wwwgate.ti.com:80
    FAST_CLONE_MODELZOO="--single-branch -b release"
else
    REPO_LOCATION=""
    PROXY_LOCATION=""
    FAST_CLONE_MODELZOO=""
fi
# print
echo "REPO_LOCATION="${REPO_LOCATION}
echo "PROXY_LOCATION="${PROXY_LOCATION}

# clone
echo "cloning git repositories. this may take some time..."
if [ -z ../edgeai-benchmark ]; then git clone ${SOURCE_LOCATION}edgeai-benchmark.git ..; fi
if [ -z ../edgeai-mmdetection ]; then git clone ${SOURCE_LOCATION}edgeai-mmdetection.git ..; fi
if [ -z ../edgeai-torchvision ]; then git clone ${SOURCE_LOCATION}edgeai-torchvision.git ..; fi
if [ -z ../edgeai-modelzoo ]; then git clone ${SOURCE_LOCATION}edgeai-modelzoo.git ${FAST_CLONE_MODELZOO} ..; fi

# Build docker image
echo "building docker image..."
docker build \
    -f Dockerfile \
    -t modelmaker \
    --build-arg REPO_LOCATION=${REPO_LOCATION} \
    --build-arg PROXY_LOCATION=${PROXY_LOCATION} \
    --build-arg USER_ID=$(id -u) \
    --build-arg USER_GID=$(id -g) \
    --no-cache .
