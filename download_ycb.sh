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

#Base archive
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_base.zip
uzip ycbv_base.zip
cd ycbv_base
#models
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip
uzip ycbv_models.zip
#train_pbr
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_pbr.zip
uzip ycbv_train_pbr.zip
#train_real
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_real.zip
uzip ycbv_train_real.zip
#All test images
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_all.zip
uzip ycbv_test_all.zip
#BOP test images
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_bop19.zip

#Run the scripts below to convert the annotations in COCO fromat.

python tools/ycb2coco.py --split train 
                         --split test                   #2949 frames for testing
                         --split test  --type bop       # 900 frames for testing as in BOP format