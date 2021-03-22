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
from jacinto_ai_benchmark import *


if __name__ == '__main__':
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    parser.add_argument('--task_selection', type=str, nargs='*')
    parser.add_argument('--model_selection', type=str, nargs='*')
    parser.add_argument('--session_type_dict', type=str, nargs='*')
    cmds = parser.parse_args()

    kwargs = dict()
    if cmds.task_selection is not None:
        kwargs.update({'task_selection':cmds.task_selection})
    #
    if cmds.model_selection is not None:
        kwargs.update({'model_selection':cmds.model_selection})
    #
    if cmds.session_type_dict is not None:
        cmds.session_type_dict = utils.str_to_dict(cmds.session_type_dict)
        kwargs.update({'session_type_dict':cmds.session_type_dict})
    #
    settings = config_settings.ConfigSettings(cmds.settings_file, **kwargs)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join('./work_dirs', expt_name, f'{settings.tidl_tensor_bits}bits')
    print(f'work_dir: {work_dir}')

    # check the datasets and download if they are missing
    download_ok = configs.download_datasets(settings)
    print(f'download_ok: {download_ok}')

    if settings.configs_path is not None:
        benchmark_configs = utils.import_folder(settings.configs_path)
        pipeline_configs = benchmark_configs.get_configs(settings, work_dir)
    else:
        pipeline_configs = None
    #

    # run the accuracy pipeline
    tools.run_accuracy(settings, work_dir, pipeline_configs)
