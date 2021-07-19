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
import tarfile
import yaml
from .. import utils, pipelines, config_settings, datasets

__all__ = ['run_model']


def run_model(settings, run_dir, pipeline_configs=None):
    # if the run_dir doesn't exist, check if tarfile exists
    tarfile_name = run_dir if run_dir.endswith('.tar.gz') else run_dir+'.tar.gz'
    run_dir = os.path.splitext(os.path.splitext(run_dir)[0])[0] if run_dir.endswith('.tar.gz') else run_dir
    if not os.path.exists(run_dir):
        if os.path.exists(tarfile_name):
            tfp = tarfile.open(tarfile_name)
            tfp.extractall(run_dir)
            tfp.close()
        #
    #
    assert os.path.exists(run_dir), f'could not find run_dir: {run_dir}'
    run_dir_base = os.path.basename(run_dir)

    # get the default configs if pipeline_configs is not given from outside
    if pipeline_configs is None:
        param_file = os.path.join(run_dir, 'param.yaml')
        with open(param_file, 'r') as rfp:
            pipeline_params = yaml.safe_load(rfp)
            pipeline_config = pipelines.create_config(pipeline_params, run_dir_base)
            model_id, session_name = run_dir_base.split('_')[:2]
            pipeline_configs = {model_id: pipeline_config}
        #
    #

    # create the pipeline_runner which will manage the sessions.
    pipeline_runner = pipelines.PipelineRunner(settings, pipeline_configs)

    # print some info
    run_dirs = [pipeline_config['session'].get_param('run_dir') for model_key, pipeline_config \
                in pipeline_runner.pipeline_configs.items()]
    run_dirs = [os.path.basename(run_dir) for run_dir in run_dirs]
    print(f'configs to run: {run_dirs}')
    print(f'number of configs: {len(pipeline_runner.pipeline_configs)}')
    sys.stdout.flush()

    # now actually run the configs
    if settings.run_import or settings.run_inference:
        pipeline_runner.run()
    #
