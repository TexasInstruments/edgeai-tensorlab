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
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}Moving to datasets${NC}"
cd datasets

echo -e "${GREEN}Downloading Base archive${NC}"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_base.zip
unzip ycbv_base.zip
rm ycbv_base.zip
cd ycbv # All other files are extracted inside this

#CAD Models
echo -e "***** ${GREEN}Downloading YCBV models ${BLUE}(524MB) ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip
echo -e "${GREEN}Extracting YCBV models${NC}"
unzip ycbv_models.zip
rm ycbv_models.zip

#Test Images
echo -e "***** ${GREEN}Downloading all test images ${BLUE}(15GB).${RED}This will take some time ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_all.zip
echo -e "${GREEN}Extracting all test images ${BLUE}(15GB)${NC}"
unzip ycbv_test_all.zip && mv test test_all  #rename test to test_all
rm ycbv_test_all.zip
wget https://raw.githubusercontent.com/yuxng/YCB_Video_toolbox/master/keyframe.txt  #seleced frames used for evaluation

echo -e "***** ${GREEN}Downloading BOP subset of test images ${BLUE}(660MB) ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_bop19.zip
echo -e "${GREEN}Extracting BOP subset of test images ${BLUE}(660MB)${NC}"
unzip ycbv_test_bop19.zip && mv test test_bop   #rename test to test_bop
ln -s test_bop test  #create a softlink of test_bop to test. This is used by the BOP evaluation script.
rm ycbv_test_bop19.zip

#Training Images
echo -e "***** ${GREEN}Downloading train_pbr subset of training images ${BLUE}(21GB).${RED}This will take some time ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_pbr.zip
echo -e "${GREEN}Extracting train_pbr subset of training images${BLUE}(21GB).${RED}This will take some time${NC}"
unzip ycbv_train_pbr.zip
rm ycbv_train_pbr.zip

echo -e "***** ${GREEN}Downloading train_real split ${BLUE}(75.7GB).${RED}This will take some time ${NC}***** "
wget https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_real.zip
echo -e "${GREEN}Extracting train_real split ${BLUE}(75.7GB).${RED}This will take some time${NC}"
unzip ycbv_train_real.zip
rm ycbv_train_real.zip

#Annotations. Placeholder for annotation files in COCO format
mkdir annotations

#Going back to edgeai-yolox
cd ../..