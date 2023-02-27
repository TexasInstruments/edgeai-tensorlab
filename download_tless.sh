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
wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_base.zip
echo  -e "***** ${GREEN}Extracting Base archive ${NC}*****"
unzip tless_base.zip
rm tless_base.zip
cd tless #All other files are extracted inside this

#CAD Models
echo  -e "***** ${GREEN}Downloading TLESS models ${BLUE}(33.5MB) ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_models.zip
echo -e "${GREEN}Extracting TLESS models ${ NC}"
unzip tless_models.zip
rm tless_models.zip

#Test Images
echo  -e "***** ${GREEN}Downloading all test images ${BLUE}(8.3GB) ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_test_primesense_all.zip
echo -e "${GREEN}Extracting all test images ${BLUE}(8.3GB)${NC}"
unzip tless_test_primesense_all.zip && mv test_primesense test_all  #rename test to test_all
rm tless_test_primesense_all.zip

echo  -e "***** ${GREEN}Downloading BOP subset of test images ${BLUE}(825.3MB)${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_test_primesense_bop19.zip
echo -e "${GREEN}Extracting BOP subset of test images ${BLUE}(825.3MB)${NC}"
unzip tless_test_primesense_bop19.zip && mv test_primesense test_bop  #rename test_primesense to test_bop
ln -s test_bop test  #create a softlink of test_bop to test. This is used by the BOP evaluation script.
rm tless_test_primesense_bop19.zip

#Training Images
echo  -e "***** ${GREEN}Downloading train_pbr subset of training images ${BLUE}(23.0GB).${RED}This will take some time ${NC}*****"
wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_train_pbr.zip
echo -e "${GREEN}Extracting train_pbr subset of training images${BLUE}(23.0GB).${RED}This will take some time${NC}"
unzip tless_train_pbr.zip
rm tless_train_pbr.zip

echo -e "***** ${GREEN}Downloading train_real split ${BLUE}(2.5GB).${RED}This will take some time ${NC}***** "
wget https://bop.felk.cvut.cz/media/data/bop_datasets/tless_train_primesense.zip
echo -e "${GREEN}Extracting train_real split ${BLUE}(2.5GB).${RED}This will take some time${NC}"
unzip tless_train_primesense.zip && mv train_primesense train_real  #rename train_primesense to train_real
rm tless_train_primesense.zip

#Annotations. Placeholder for annotation files in COCO format
mkdir annotations

#Going back to edgeai-yolox
cd ../..