#!/usr/bin/env bash

# Copyright (c) 2023-2024, Texas Instruments Incorporated
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

# note: this script is for development only - not meant to be used by users

RELEASE_BRANCH=${1-"r11.0"}

GIT_REPOS=( edgeai-modelzoo edgeai-modelmaker edgeai-modeloptimization edgeai-benchmark edgeai-torchvision edgeai-tensorvision edgeai-mmdetection edgeai-mmdetection3d edgeai-mmpose edgeai-hf-transformers edgeai-mmdeploy edgeai-yolox )

# removed repositories
# edgeai-modelutils edgeai-datasets edgeai-mmrazor edgeai-docs

for GIT_REPO in "${GIT_REPOS[@]}"
do
  git subtree add --prefix ${GIT_REPO} ssh://git@bitbucket.itg.ti.com/edgeai-algo/${GIT_REPO}.git ${RELEASE_BRANCH}
  git subtree pull --prefix ${GIT_REPO} ssh://git@bitbucket.itg.ti.com/edgeai-algo/${GIT_REPO}.git ${RELEASE_BRANCH}
done

#cd edgeai-mlbackend
#git pull --rebase
#cd ..
