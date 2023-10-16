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

import os
import argparse
import yaml

from edgeai_benchmark import *


if __name__ == '__main__':
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--target_device', type=str)
    parser.add_argument('--param_template_file', type=str, default='./examples/configs/yaml/param_template.yaml')

    cmds = parser.parse_args()
    #kwargs = vars(cmds)

    settings = config_settings.ConfigSettings(cmds.settings_file, target_device=cmds.target_device)

    param_template = None
    if cmds.param_template_file is not None:
        with open(cmds.param_template_file) as fp:
            param_template = yaml.safe_load(fp)
        #
    #

    if 'TIDL_ARTIFACT_SYMLINKS' in os.environ and os.environ['TIDL_ARTIFACT_SYMLINKS']:
        if cmds.work_dir is None:
            work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
        else:
            work_dir = cmds.work_dir
        print(f'work_dir: {work_dir}')

        if cmds.out_dir is None:
            package_dir = settings.modelpackage_path
            out_dir = os.path.join(package_dir, f'{settings.tensor_bits}bits')
        else:
            out_dir = cmds.out_dir
        print(f'package_dir: {out_dir}')

        interfaces.run_package(settings, work_dir, out_dir, param_template=param_template)
    else:
        print('TIDL_ARTIFACT_SYMLINKS is not set - run this script using run_package_artifacts_for_evm.sh')
    #

