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

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    cmds = parser.parse_args()
    settings = config_settings.ConfigSettings(cmds.settings_file)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join('./work_dirs', expt_name, f'{settings.tidl_tensor_bits}bits')
    print(f'work_dir: {work_dir}')

    # check the datasets and download if they are missing
    download_ok = configs.download_datasets(settings)
    print(f'download_ok: {download_ok}')

    # get the default configs available
    pipeline_configs = configs.get_configs(settings, work_dir)

    # create the pipeline_runner which will manage the sessions.
    pipeline_runner = pipelines.PipelineRunner(settings, pipeline_configs)

    ############################################################################
    # at this point, pipeline_runner.pipeline_configs is a dictionary that has the selected configs

    # Note: to manually slice and select a subset of configs, slice it this way (just an example)
    # import itertools
    # pipeline_runner.pipeline_configs = dict(itertools.islice(pipeline_runner.pipeline_configs.items(), 10, 20))

    # some examples of accessing params from it - here 0th entry is used an example.
    # pipeline_config = pipeline_runner.pipeline_configs.values()[0]
    # pipeline_config['preprocess'].get_param('resize') gives the resize dimension
    # pipeline_config['preprocess'].get_param('crop') gives the crop dimension
    # pipeline_config['session'].get_param('run_dir') gives the folder where artifacts are located
    ############################################################################

    # print some info
    run_dirs = [pipeline_config['session'].get_param('run_dir') for model_key, pipeline_config \
                in pipeline_runner.pipeline_configs.items()]
    run_dirs = [os.path.basename(run_dir) for run_dir in run_dirs]
    print(f'configs to run: {run_dirs}')
    print(f'number of configs: {len(pipeline_runner.pipeline_configs)}')

    # now actually run the configs
    if settings.run_import or settings.run_inference:
        pipeline_runner.run()
    #

    # save params in a yaml file - also requires enable_logging to be set.
    # if enable_logging is False, nothing will be written to file
    if settings.save_params and settings.enable_logging:
        pipelines.save_params(settings, work_dir, pipeline_runner.pipeline_configs)
    #

    # collect the logs and display it
    # requires enable_logging to be True to write results to file
    if settings.collect_results:
        results = pipelines.collect_results(settings, work_dir, pipeline_runner.pipeline_configs, print_results=True)
    #

