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
import functools
from jacinto_ai_benchmark import *

# the PYTHONPATH must start with a : or .: for this line to work
import benchmark_accuracy

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#


def modify_pipelines(cmds, pipeline_configs):
    if cmds.input_size is not None:
        # modify a pipeline so that all the models use fixed input size
        # other modifications can also be defined here.
        warning_string = f'Changing input size to {cmds.input_size}.\n' \
                         f'The accuracies reported may be wrong as input_size is changed from the default value.'
        print(warning_string)
        for pipeline_id, pipeline_config in pipeline_configs.items():
            preproc_stge = pipeline_config['preprocess']
            preproc_transforms = preproc_stge.transforms
            for tidx, trans in enumerate(preproc_transforms):
                if isinstance(trans, preprocess.ImageResize):
                    trans = preprocess.ImageResize(cmds.input_size)
                elif isinstance(trans, preprocess.ImageCenterCrop):
                    trans = preprocess.ImageCenterCrop(cmds.input_size)
                #
                preproc_transforms[tidx] = trans
            #
        #
    #
    return pipeline_configs


if __name__ == '__main__':
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    parser.add_argument('--input_size', type=int, default=None)
    cmds = parser.parse_args()

    # for performance measurement, we need to use only one frame
    settings = config_settings.ConfigSettings(cmds.settings_file,
            num_frames=1, num_frames_calib=1, max_calib_iterations=1)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join('./work_dirs', expt_name, f'{settings.tidl_tensor_bits}bits')
    print(f'work_dir: {work_dir}')

    modify_pipelines_func = functools.partial(modify_pipelines, cmds)
    benchmark_accuracy.main(settings, work_dir, modify_pipelines_func=modify_pipelines_func)
