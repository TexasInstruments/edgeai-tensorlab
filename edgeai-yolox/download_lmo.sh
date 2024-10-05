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
NC='\033[0m'
BLUE='\033[0;34m'

echo  -e "${GREEN}Moving to datasets${NC}"
cd datasets

echo  -e "***** ${GREEN}Downloading Base archive ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_base.zip
echo  -e "***** ${GREEN}Extracting Base archive ${NC}*****"
unzip lmo_base.zip
rm lmo_base.zip
cd lmo #All other files are extracted inside this

#CAD Models
echo  -e "***** ${GREEN}Downloading LM models ${BLUE}(5.4MB) ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_models.zip
echo -e "${GREEN}Extracting LM models${NC}"
unzip lmo_models.zip
rm lmo_models.zip

#Test Images
echo  -e "***** ${GREEN}Downloading all test images ${BLUE}(720.2MB) ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_test_all.zip
echo -e "${GREEN}Extracting all test images ${BLUE}(720.2MB)${NC}"
unzip lmo_test_all.zip && mv test test_all  #rename test to test_all
rm lmo_test_all.zip

echo  -e "***** ${GREEN}Downloading BOP subset of test images ${BLUE}(117.6MB)${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_test_bop19.zip
echo -e "${GREEN}Extracting BOP subset of test images ${BLUE}(117.6MB)${NC}"
unzip lmo_test_bop19.zip && mv test test_bop  #rename test to test_bop
ln -s test_bop test  #create a softlink of test_bop to test. This is used by the BOP evaluation script.
rm lmo_test_bop19.zip

#Training Images
echo  -e "***** ${GREEN}Downloading train_pbr subset of training images ${BLUE}(21.8GB).${RED}This will take some time ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/lm_train_pbr.zip
echo -e "${GREEN}Extracting train_pbr subset of training images${BLUE}(21.8GB).${RED}This will take some time${NC}"
unzip lm_train_pbr.zip
rm lm_train_pbr.zip

#Annotations. Placeholder for annotation files in COCO format
mkdir annotations

#Going back to edgeai-yolox
cd ../..