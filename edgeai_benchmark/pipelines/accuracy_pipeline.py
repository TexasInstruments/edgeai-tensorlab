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
import copy
import yaml
import time
import itertools
from .. import utils, constants
from .base_pipeline import BasePipeline


class AccuracyPipeline(BasePipeline):
    def __init__(self, settings, pipeline_config):
        super().__init__(settings, pipeline_config)

    def __call__(self, description=''):
        ##################################################################
        # check and return if result exists
        if self.settings.run_incremental and os.path.exists(self.result_yaml):
            with open(self.result_yaml) as fp:
                param_result = yaml.safe_load(fp)
                result_dict = param_result['result'] if 'result' in param_result else {}
            #
            if self.settings.rewrite_results and self.settings.write_results:
                param_dict = utils.pretty_object(self.pipeline_config)
                with open(self.param_yaml, 'w') as fp:
                    yaml.safe_dump(param_dict, fp, sort_keys=False)
                #
                param_result = dict({'result': result_dict})
                param_result.update(param_dict)
                with open(self.result_yaml, 'w') as fp:
                    yaml.safe_dump(param_result, fp, sort_keys=False)
                #
            #
            print(utils.log_color('\nSUCCESS', 'found results', f'{result_dict}\n'))
            return param_result
        #

        ##################################################################
        # start() must be called to create the required directories
        self.session.start()

        # start logger - run_dir has been created in start() above
        log_filename = os.path.join(self.run_dir, 'run.log') if self.settings.enable_logging else None
        logger_buffering = (1 if self.settings.capture_log else -1)
        self.logger = utils.TeeLogger(log_filename, mode='a', buffering=logger_buffering)

        # log some info
        self.write_log(utils.log_color('\nINFO', 'running', os.path.basename(self.run_dir)))
        self.write_log(utils.log_color('\nINFO', 'pipeline_config', self.pipeline_config))

        # write the dataset_info file
        if self.settings.write_results and self.dataset_info is not None:
            with open(self.dataset_info_file, 'w') as fp:
                yaml.safe_dump(self.dataset_info, fp, sort_keys=False)
            #
        #

        # now actually run the import and inference
        param_result = self._run(description=description)

        result_dict = param_result.get('result', {})
        self.write_log(utils.log_color('\n\nSUCCESS', 'benchmark results', f'{result_dict}\n'))
        self.logger.close()
        self.logger = None
        return param_result

    def _run(self, description=''):
        param_result = {}

        ##################################################################
        # import.
        run_import = ((not os.path.exists(self.param_yaml)) if self.settings.run_incremental else True) \
            if self.settings.run_import else False
        if run_import:
            # dump the config params
            if self.settings.write_results:
                param_dict = utils.pretty_object(self.pipeline_config)
                with open(self.config_yaml, 'w') as fp:
                    yaml.safe_dump(param_dict, fp, sort_keys=False)
                #
            #

            start_time = time.time()
            self.write_log(utils.log_color('\nINFO', f'import {description}', self.run_dir_base + ' - this may take some time...'))
            # import stats
            self._import_model(description)
            elapsed_time = time.time() - start_time
            self.write_log(utils.log_color('\nINFO', f'import completed {description}', f'{self.run_dir_base} - {elapsed_time:.0f} sec'))

            # collect the input params
            param_dict = utils.pretty_object(self.pipeline_config)
            param_result = param_dict

            # dump the params after import
            if self.settings.write_results:
                with open(self.param_yaml, 'w') as fp:
                    yaml.safe_dump(param_dict, fp, sort_keys=False)
                #
                with open(self.config_yaml, 'w') as fp:
                    yaml.safe_dump(param_dict, fp, sort_keys=False)
                #
            #
        #

        ##################################################################
        # inference
        if self.settings.run_inference:
            start_time = time.time()
            self.write_log(utils.log_color('\nINFO', f'infer {description}', self.run_dir_base + ' - this may take some time...'))
            output_list = self._infer_frames(description)
            elapsed_time = time.time() - start_time
            self.write_log(utils.log_color('\nINFO', f'infer completed {description}', f'{self.run_dir_base} - {elapsed_time:.0f} sec'))
            result_dict = self._evaluate(output_list)
            # collect the results
            result_dict.update(self.infer_stats_dict)
            result_dict = utils.pretty_object(result_dict)
            # collect the params once again, as it might have changed internally
            param_dict = utils.pretty_object(self.pipeline_config)
            param_result = dict(result=result_dict, **param_dict)
            # dump the results
            if self.settings.write_results:
                with open(self.result_yaml, 'w') as fp:
                    yaml.safe_dump(param_result, fp, sort_keys=False)
                #
            #
        #
        return param_result

    def _import_model(self, description=''):
        session = self.pipeline_config['session']
        calibration_dataset = self.pipeline_config['calibration_dataset']
        assert calibration_dataset is not None, f'got input_dataset={calibration_dataset}. please check settings.dataset_loading'
        preprocess = self.pipeline_config['preprocess']
        runtime_options = self.pipeline_config['session'].peek_param('runtime_options')
        calibration_frames = runtime_options['advanced_options']['calibration_frames'] \
            if 'advanced_options' in runtime_options else runtime_options['advanced_options:calibration_frames']
        assert len(calibration_dataset) >= calibration_frames, \
            utils.log_color('\nERROR', 'import', f'too few calibration data - calibration dataset size ({len(calibration_dataset)}) '
                                                 f'should be >= calibration_frames ({calibration_frames})')

        calib_data = []
        for data_index in range(calibration_frames):
            info_dict = {'dataset_info': self.dataset_info, 'label_offset_pred': self.pipeline_config.get('metric',{}).get('label_offset_pred',None)}
            data = calibration_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)
            calib_data.append(data)
        #

        # this is the actual import
        self._run_with_log(session.import_model, calib_data)
        # close the interpreter
        session.close_interpreter()

    def _infer_frames(self, description=''):
        session = self.pipeline_config['session']
        input_dataset = self.pipeline_config['input_dataset']
        assert input_dataset is not None, f'got input_dataset={input_dataset}. please check settings.dataset_loading'
        preprocess = self.pipeline_config['preprocess']
        postprocess = self.pipeline_config['postprocess']
        run_dir_base = os.path.split(session.get_param('run_dir'))[-1]
        num_frames = self.pipeline_config.get('num_frames', self.settings.num_frames)
        num_frames = min(len(input_dataset), num_frames) if num_frames else len(input_dataset)

        is_ok = session.start_infer()
        assert is_ok, utils.log_color('\nERROR', f'start_infer() did not succeed for:', run_dir_base)

        invoke_time = 0.0
        core_time = 0.0
        subgraph_time = 0.0
        ddr_transfer = 0.0
        num_frames_ddr = 0

        output_list = []
        pbar_desc = f'infer {description}: {run_dir_base}'
        for data_index in utils.progress_step(range(num_frames), desc=pbar_desc, file=self.logger, position=0):
            info_dict = {'dataset_info': self.dataset_info, 'label_offset_pred': self.pipeline_config.get('metric',{}).get('label_offset_pred',None)}
            data = input_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)
            output, info_dict = self._run_with_log(session.infer_frame, data, info_dict)
            invoke_time += info_dict['session_invoke_time']

            stats_dict = session.infer_stats()
            core_time += stats_dict['core_time']
            subgraph_time += stats_dict['subgraph_time']
            if stats_dict['write_total'] >= 0  and stats_dict['read_total'] >= 0 :
                ddr_transfer += (stats_dict['write_total'] + stats_dict['read_total'])
                num_frames_ddr += 1
           
            if self.settings.flip_test:
                outputs_flip, info_dict = self._run_with_log(session.infer_frame, info_dict['flip_img'], info_dict)
                info_dict['outputs_flip'] = outputs_flip
                invoke_time += info_dict['session_invoke_time']

                stats_dict = session.infer_stats()
                core_time += stats_dict['core_time']
                subgraph_time += stats_dict['subgraph_time']
                if stats_dict['write_total'] >= 0  and stats_dict['read_total'] >= 0 :
                    ddr_transfer += (stats_dict['write_total'] + stats_dict['read_total'])
                    num_frames_ddr += 1
            else:
                info_dict['outputs_flip'] = None

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
        # close the interpreter
        session.close_interpreter()
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

    def _run_with_log(self, func, *args, **kwargs):
        log_fp = self.logger.log_file if self.logger is not None else None
        logging_mode = 'wurlitzer' if self.settings.capture_log else None
        if log_fp is None or logging_mode is None:
            return func(*args, **kwargs)
        elif logging_mode == 'redirect_logger':
            # redirect prints to file using os.dup2()
            # observation: may not work well with multiprocessing
            with utils.RedirectLogger(log_fp):
                return func(*args, **kwargs)
            #
        elif logging_mode == 'wurlitzer':
            # redirect logs using wurlitzer
            # this works well with multiprocessing, but causes the execution to slow down
            import wurlitzer
            with wurlitzer.pipes(stdout=log_fp, stderr=wurlitzer.STDOUT):
                return func(*args, **kwargs)
            #
        else:
            assert False, f'_run_with_log: unknown logging_level {logging_mode}'
        #
