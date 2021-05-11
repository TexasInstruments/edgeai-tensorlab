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
import pickle
import yaml
from colorama import Fore
import wurlitzer
from .. import utils, constants


class AccuracyPipeline():
    def __init__(self, settings, pipeline_config):
        self.info_dict = dict()
        self.settings = settings
        self.pipeline_config = pipeline_config
        self.avg_inference_time = None
        self.logger = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, description=''):
        session = self.pipeline_config['session']
        run_dir = session.get_param('run_dir')
        # create logfile if required.
        if self.settings.enable_logging:
            log_filename = os.path.join(run_dir, 'run.log')
            os.makedirs(run_dir, exist_ok=True)
            log_fp = open(log_filename, 'w')
        else:
            log_fp = None
        #
        # start logger and run the pipeline
        self.logger = utils.TeeLogger(log_fp)
        param_result = self._run(description=description)
        self.logger.close()
        self.logger = None
        return param_result

    def _run(self, description=''):
        # initialize result as empty
        result_dict = {}
        # run the actual model
        session = self.pipeline_config['session']
        # run_dir is assigned after initialize is called in PipelineRunner
        # but it has not been created - it will be created in start
        run_dir = session.get_param('run_dir')
        run_dir_base = os.path.split(run_dir)[-1]
        # check if the result already exists - if so we can return
        result_yaml = os.path.join(run_dir, 'result.yaml')
        if self.settings.run_missing and os.path.exists(result_yaml):
            with open(result_yaml) as fp:
                param_result = yaml.safe_load(fp)
                result_dict = param_result['result'] if 'result' in param_result else {}
            #
            if self.settings.rewrite_results and self.settings.enable_logging:
                param_dict = utils.pretty_object(self.pipeline_config)
                param_result = dict({'result': result_dict})
                param_result.update(param_dict)
                with open(os.path.join(run_dir, 'result.yaml'), 'w') as fp:
                    yaml.safe_dump(param_result, fp, sort_keys=False)
                #
            #
            return param_result
        else:
            # collect the input params
            param_dict = utils.pretty_object(self.pipeline_config)
            # start() must be called to create the required directories
            session.start()
            # log some info
            self.logger.write(utils.log_color('\nINFO', 'running', os.path.basename(run_dir)))
            self.logger.write(utils.log_color('\nINFO', 'pipeline_config', self.pipeline_config))
            # import.
            if self.settings.run_import:
                self.logger.write(utils.log_color('\nINFO', f'import & calibration {description}', run_dir_base))
                self._run_with_log(self.logger.log_file, self._import_model, description)
                self.logger.write(utils.log_color('\nINFO', f'import & calibration {description}', f'{run_dir_base} - done'))
            #
            # inference
            if self.settings.run_inference:
                self.logger.write(utils.log_color('\nINFO', f'infer {description}', run_dir_base))
                output_list = self._run_with_log(self.logger.log_file, self._infer_frames, description)
                self.logger.write(utils.log_color('\nINFO', f'infer {description}', f'{run_dir_base} - done.'))
                result_dict = self._evaluate(output_list)
                result_dict.update(self.infer_stats_dict)
            #
            # collect the results
            result_dict = utils.pretty_object(result_dict)
            param_result = dict({'result': result_dict})
            param_result.update(param_dict)
            # dump the results
            if self.settings.enable_logging:
                with open(os.path.join(run_dir, 'result.yaml'), 'w') as fp:
                    yaml.safe_dump(param_result, fp, sort_keys=False)
                #
            #
            self.logger.write(utils.log_color('\nSUCCESS', 'benchmark results', f'{result_dict}\n'))
            return param_result
        #

    def _import_model(self, description=''):
        session = self.pipeline_config['session']
        calibration_dataset = self.pipeline_config['calibration_dataset'] \
            if 'calibration_dataset' in self.pipeline_config else None
        if calibration_dataset is None:
            calibration_dataset = self.pipeline_config['input_dataset']
        #
        assert calibration_dataset is not None, f'got input_dataset={calibration_dataset}. please check settings.dataset_loading'
        preprocess = self.pipeline_config['preprocess']

        calib_data = []
        calibration_frames = min(len(calibration_dataset), self.settings.calibration_frames)
        for data_index in range(calibration_frames):
            info_dict = {}
            data = calibration_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)
            calib_data.append(data)
        #
        session.import_model(calib_data)

    def _infer_frames(self, description=''):
        session = self.pipeline_config['session']
        input_dataset = self.pipeline_config['input_dataset']
        assert input_dataset is not None, f'got input_dataset={input_dataset}. please check settings.dataset_loading'
        preprocess = self.pipeline_config['preprocess']
        postprocess = self.pipeline_config['postprocess']
        run_dir_base = os.path.split(session.get_param('run_dir'))[-1]

        is_ok = session.start_infer()
        assert is_ok, utils.log_color('\nERROR', f'start_infer() did not succeed for', run_dir_base)

        invoke_time = 0.0
        core_time = 0.0
        subgraph_time = 0.0
        ddr_transfer = 0.0
        num_frames_ddr = 0

        output_list = []
        num_frames = min(len(input_dataset), self.settings.num_frames)
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
            if stats_dict['write_total'] >= 0  and stats_dict['read_total'] >= 0 :
                ddr_transfer += (stats_dict['write_total'] + stats_dict['read_total'])
                num_frames_ddr += 1

            output, info_dict = postprocess(output, info_dict)
            output_list.append(output)
        #
        # compute and populate final stats so that it can be used in result
        self.infer_stats_dict = {
            'num_subgraphs': stats_dict['num_subgraphs'],
            #'infer_time_invoke_ms': invoke_time * constants.MILLI_CONST / num_frames,
            'infer_time_core_ms': core_time * constants.MILLI_CONST / num_frames,
            'infer_time_subgraph_ms': subgraph_time * constants.MILLI_CONST / num_frames,
            'ddr_transfer_mb': (ddr_transfer / num_frames_ddr / constants.MEGA_CONST) if num_frames_ddr > 0 else 0
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

    def _run_with_log(self, log_fp, func, *args, **kwargs):
        if log_fp is not None:
            with wurlitzer.pipes(stdout=log_fp, stderr=wurlitzer.STDOUT):
                return func(*args, **kwargs)
            #
        else:
            return func(*args, **kwargs)
