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

DOCKER_IMAGE_NAME="edgeai-modelmaker:10.1.0"
DOCKER_CONTAINER_NAME="edgeai-modelmaker-10.1.0-cnt"
PARENT_DIR=$(realpath ..)

# This script is intended to work with single container.
# Number container exist
container_count=$(docker ps -a | grep ${DOCKER_CONTAINER_NAME} | wc -l)
echo "Number of containers with the given name/tag: ${container_count} "

echo "This script starts the container with GPU support."
echo "Make sure you have installed GPUs, nvidia drivers and also nvidia-docker2."

# initialize http_proxy and https_proxy if they are not defined
http_proxy=${http_proxy:-""}
https_proxy=${https_proxy:-""}
no_proxy=${no_proxy:-""}

#If no container exist, then create the container.
if [ $container_count -eq 0 ]
then
    echo "Starting a new container: ${DOCKER_CONTAINER_NAME}"
    docker run -it \
        --name "${DOCKER_CONTAINER_NAME}" \
        -v ${PARENT_DIR}:/opt/code \
        --privileged \
        --network host \
        --shm-size 50G \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -e no_proxy=${no_proxy} \
        --user $(id -u):$(id -g) \
        ${DOCKER_IMAGE_NAME} bash
# If one container exist, execute that container.
elif [ $container_count -eq 1 ]
then
    echo "Restarting existing container: ${DOCKER_CONTAINER_NAME}"
    docker start "${DOCKER_CONTAINER_NAME}"
    docker exec -it "${DOCKER_CONTAINER_NAME}" /bin/bash
else
    echo -e "\nMultiple containers found with similar name/tag ${DOCKER_CONTAINER_NAME}, so exiting"
    echo -e "To run existing container, use [docker start] and [docker exec] command"
    echo -e "To run the new container, use [docker run] command\n"
fi
