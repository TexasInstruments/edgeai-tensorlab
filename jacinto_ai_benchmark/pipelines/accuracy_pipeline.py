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
import pickle
import yaml
from colorama import Fore
from .. import utils, constants


class AccuracyPipeline():
    def __init__(self, settings, pipeline_config):
        self.info_dict = dict()
        self.settings = settings
        self.pipeline_config = pipeline_config
        self.logger = None
        self.avg_inference_time = None

    def __del__(self):
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        #

    def run(self, description=''):
        # run the actual model
        result = {}
        run_import = self.pipeline_config['run_import']
        run_inference = self.pipeline_config['run_inference']
        session = self.pipeline_config['session']

        # start() must be called to create the required directories
        session.start()

        # logger can be created after start
        run_dir = session.get_param('run_dir')
        # verbose = self.pipeline_config['verbose']
        log_filename = os.path.join(run_dir, 'run.log') if self.settings.enable_logging else None
        self.logger = utils.TeeLogger(log_filename)
        self.logger.write(f'\nrunning: {Fore.BLUE}{os.path.basename(run_dir)}{Fore.RESET}')
        self.logger.write(f'\npipeline_config: {self.pipeline_config}')

        if run_import:
            self._import_model(description)
        #
        if run_inference:
            output_list = self._infer_frames(description)
            result = self._evaluate(output_list)
            result.update(self.infer_stats_dict)
        #
        self.logger.write(f'\nBenchmarkResults: {utils.round_dict(result)}\n')

        if self.settings.enable_logging:
            yaml_filename = os.path.join(run_dir, 'result.yaml')
            with open(yaml_filename, 'w') as fp:
                yaml.safe_dump(result, fp)
            #
            # TODO: deprecate this pkl file format later
            pkl_filename = os.path.join(run_dir, 'result.pkl')
            with open(pkl_filename, 'wb') as fp:
                pickle.dump(result, fp)
            #
        #
        return result

    def _import_model(self, description=''):
        session = self.pipeline_config['session']
        calibration_dataset = self.pipeline_config['calibration_dataset']
        assert calibration_dataset is not None, f'got input_dataset={calibration_dataset}. please check settings.dataset_loading'
        preprocess = self.pipeline_config['preprocess']
        run_dir_base = os.path.split(session.get_param('run_dir'))[-1]

        self.logger.write(f'\nimport & calibration {description}:' + run_dir_base)
        calib_data = []
        num_frames = len(calibration_dataset)
        for data_index in range(num_frames):
            info_dict = {}
            data = calibration_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)
            calib_data.append(data)
        #
        session.import_model(calib_data)
        self.logger.write(f'\nimport & calibration {description}: {run_dir_base} - done.')

    def _infer_frames(self, description=''):
        session = self.pipeline_config['session']
        input_dataset = self.pipeline_config['input_dataset']
        assert input_dataset is not None, f'got input_dataset={input_dataset}. please check settings.dataset_loading'
        preprocess = self.pipeline_config['preprocess']
        postprocess = self.pipeline_config['postprocess']
        run_dir_base = os.path.split(session.get_param('run_dir'))[-1]

        is_ok = session.start_infer()
        assert is_ok, f'start_infer() did not succeed for {run_dir_base}'

        self.logger.write(f'\ninfer {description}:' + run_dir_base)

        invoke_time = 0.0
        core_time = 0.0
        subgraph_time = 0.0
        ddr_transfer = 0.0

        output_list = []
        num_frames = len(input_dataset)
        pbar_desc = f'infer {description}: {run_dir_base}'
        for data_index in utils.progress_step(range(num_frames), desc=pbar_desc, file=self.logger, position=0):
            info_dict = {}
            data = input_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)

            output, info_dict = session.infer_frame(data, info_dict)
            invoke_time += info_dict['session_invoke_time']

            stats_dict = session.infer_stats()
            core_time += stats_dict['core_time']
            subgraph_time += stats_dict['subgraph_time']
            ddr_transfer += (stats_dict['write_total'] + stats_dict['read_total'])

            output, info_dict = postprocess(output, info_dict)
            output_list.append(output)
        #
        # compute and populate final stats so that it can be used in result
        self.infer_stats_dict = {
            'num_subgraphs': stats_dict['num_subgraphs'],
            #'infer_time_invoke_ms': invoke_time * constants.MILLI_CONST / num_frames,
            'infer_time_core_ms': core_time * constants.MILLI_CONST / num_frames,
            'infer_time_subgraph_ms': subgraph_time * constants.MILLI_CONST / num_frames,
            'ddr_transfer_mb': ddr_transfer / num_frames / constants.MEGA_CONST
        }
        if 'perfsim_time' in stats_dict:
            self.infer_stats_dict.update({'perfsim_time_ms': stats_dict['perfsim_time'] * constants.MILLI_CONST})
        #
        if 'perfsim_ddr_transfer' in stats_dict:
            self.infer_stats_dict.update({'perfsim_ddr_transfer_mb': stats_dict['perfsim_ddr_transfer'] / constants.MEGA_CONST})
        #
        if 'perfsim_macs' in stats_dict:
            self.infer_stats_dict.update({'perfsim_gmacs': stats_dict['perfsim_macs'] / constants.GIGA_CONST})
        #
        self.logger.write(f'\ninfer {description}: {run_dir_base} - done.')
        return output_list

    def _evaluate(self, output_list):
        session = self.pipeline_config['session']
        # if metric is not given use input_dataset
        if 'metric' in self.pipeline_config and callable(self.pipeline_config['metric']):
            metric = self.pipeline_config['metric']
            metric_options = {}
        else:
            metric = self.pipeline_config['input_dataset']
            metric_options = self.pipeline_config.get('metric', {})
        #
        run_dir = session.get_param('run_dir')
        metric_options['run_dir'] = run_dir
        metric = utils.as_list(metric)
        metric_options = utils.as_list(metric_options)
        output_dict = {}
        inference_path = os.path.split(run_dir)[-1]
        output_dict.update({'infer_path':inference_path})
        for m, m_options in zip(metric, metric_options):
            output = m(output_list, **m_options)
            output_dict.update(output)
        #
        return output_dict
