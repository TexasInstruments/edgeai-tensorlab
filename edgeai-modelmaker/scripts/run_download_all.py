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
import os
import datetime
import sys
import argparse
import yaml
import json


def run(config):
    import edgeai_modelmaker
    from scripts import run_generate_description

    description, help_descriptions = run_generate_description.run(config)
    model_descriptions = description['model_descriptions']
    sample_dataset_descriptions = description['sample_dataset_descriptions']

    for model_description_key, model_description in model_descriptions.items():
        model_description = edgeai_modelmaker.utils.ConfigDict(model_description)
        edgeai_modelmaker.utils.download_all(model_description)
    #

    for sample_dataset_description_key, sample_dataset_description in sample_dataset_descriptions.items():
        sample_dataset_description = edgeai_modelmaker.utils.ConfigDict(sample_dataset_description)
        # add download_path from commandline
        sample_dataset_description.update(config)
        edgeai_modelmaker.utils.download_all(sample_dataset_description)
    #
    print('\nSUCCESS: ModelMaker - Download completed.')
    return True


def main(args):
    # override with supported commandline args
    kwargs = vars(args)
    config = dict(common=dict(), dataset=dict())
    if 'target_module' in kwargs:
        config['common']['target_module'] = kwargs['target_module']
    #
    if 'download_path' in kwargs:
        config['common']['download_path'] = kwargs['download_path']
    #

    run(config)


if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the repository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--target_module', type=str, default='vision')
    parser.add_argument('--download_path', type=str, default='./data/downloads')
    args = parser.parse_args()

    main(args)
