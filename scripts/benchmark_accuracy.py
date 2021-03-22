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
import sys
import argparse
from jacinto_ai_benchmark import *


if __name__ == '__main__':
    print(f'argv={sys.argv}')
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str, default=None)
    parser.add_argument('--configs_path', type=str, default=None)
    parser.add_argument('--modelzoo_path', type=str, default=None)
    parser.add_argument('--task_selection', type=str, nargs='*', default=None)
    parser.add_argument('--model_selection', type=str, nargs='*', default=None)
    parser.add_argument('--session_type_dict', type=str, nargs='*', default=None)
    cmds = parser.parse_args()

    # this update condition is so that settings is not changed
    # if an empty string is given from the commandline
    # this makes it easy to select a suitable setting in the shell script
    dict_update_condition = lambda x:(x not in (None,''))
    kwargs = utils.dict_update_conditional({}, condition_fn=dict_update_condition,
                configs_path=cmds.configs_path, modelzoo_path=cmds.modelzoo_path,
                task_selection=cmds.task_selection, model_selection=cmds.model_selection,
                session_type_dict=utils.str_to_dict(cmds.session_type_dict))
    settings = config_settings.ConfigSettings(cmds.settings_file, **kwargs)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join('./work_dirs', expt_name, f'{settings.tidl_tensor_bits}bits')
    print(f'work_dir: {work_dir}')

    # run the accuracy pipeline
    tools.run_accuracy(settings, work_dir)
