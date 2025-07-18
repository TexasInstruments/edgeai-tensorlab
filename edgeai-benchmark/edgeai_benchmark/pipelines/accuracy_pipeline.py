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
import queue
import numpy as np
from .. import utils, constants
from .base_pipeline import BasePipeline


class AccuracyPipeline(BasePipeline):
    def __init__(self, settings, pipeline_config):
        super().__init__(settings, pipeline_config)
        self.queue_mem = None
        self.queue = None

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
        # log some info
        self.write_log(utils.log_color('\nINFO', 'running', os.path.basename(self.run_dir)))
        self.write_log(utils.log_color('\nINFO', 'pipeline_config', self.pipeline_config))

        # now actually run the import and inference
        param_result = self._run(description=description)

        result_dict = param_result.get('result', {})
        self.write_log(utils.log_color('\n\nSUCCESS', 'benchmark results', f'{result_dict}\n'))
        return param_result

    def _run(self, description=''):
        param_result = {}

        ##################################################################
        # import.
        run_import = ((not os.path.exists(self.param_yaml)) if self.settings.run_incremental else True) \
            if self.settings.run_import else False
        if run_import:
            # write the dataset_info file
            if self.settings.write_results and self.dataset_info is not None:
                with open(self.dataset_info_file, 'w') as fp:
                    yaml.safe_dump(self.dataset_info, fp, sort_keys=False)
                #
            #

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
            if self.pipeline_config['task_type'] != 'bev_detection':
                self._import_model(description)
            else:
                self._import_bev_model(description)
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
            if self.pipeline_config['task_type'] != 'bev_detection':
                output_list = self._infer_frames(description)
            else:
                output_list = self._infer_bev_frames(description)
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
        run_dir_base = os.path.split(session.get_param('run_dir'))[-1]

        is_ok = session.start_import()
        assert is_ok, utils.log_color('\nERROR', f'start_import() did not succeed for:', run_dir_base)

        for data_index in range(calibration_frames):
            info_dict = {'dataset_info': self.dataset_info, 'label_offset_pred': self.pipeline_config.get('metric',{}).get('label_offset_pred',None)}
            input_data = calibration_dataset[data_index]
            input_data, info_dict = preprocess(input_data, info_dict)
            # this is the actual import
            output, info_dict = session.run_import(input_data, info_dict)
        #

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

        is_ok = session.start_inference()
        assert is_ok, utils.log_color('\nERROR', f'start_infer() did not succeed for:', run_dir_base)

        if self.settings.target_machine == constants.TARGET_MACHINE_EVM:
            invoke_time = 0.0
            core_time = 0.0
            subgraph_time = 0.0
            ddr_transfer = 0.0
            num_frames_ddr = 0
        #

        output_list = []
        pbar_desc = f'infer {description}: {run_dir_base}'
        for data_index in utils.progress_step(range(num_frames), desc=pbar_desc, position=0):
            info_dict = {'dataset_info': self.dataset_info, 'label_offset_pred': self.pipeline_config.get('metric',{}).get('label_offset_pred',None)}
            data = input_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)
            output, info_dict = session.run_inference(data, info_dict)

            stats_dict = session.infer_stats()
            if self.settings.target_machine == constants.TARGET_MACHINE_EVM:
                invoke_time += info_dict['session_invoke_time']
                core_time += stats_dict['core_time']
                subgraph_time += stats_dict['subgraph_time']
                if stats_dict['write_total'] >= 0  and stats_dict['read_total'] >= 0 :
                    ddr_transfer += (stats_dict['write_total'] + stats_dict['read_total'])
                    num_frames_ddr += 1
               #
           #
            if self.settings.flip_test:
                outputs_flip, info_dict = session.run_inference(info_dict['flip_img'], info_dict)
                info_dict['outputs_flip'] = outputs_flip

                stats_dict = session.infer_stats()
                if self.settings.target_machine == constants.TARGET_MACHINE_EVM:
                    invoke_time += info_dict['session_invoke_time']
                    core_time += stats_dict['core_time']
                    subgraph_time += stats_dict['subgraph_time']
                    if stats_dict['write_total'] >= 0  and stats_dict['read_total'] >= 0 :
                        ddr_transfer += (stats_dict['write_total'] + stats_dict['read_total'])
                        num_frames_ddr += 1
                    #
                #
            else:
                info_dict['outputs_flip'] = None
            #

            # needed in postprocess to understand the detection threshold set
            info_dict['runtime_options'] = session.kwargs['runtime_options']

            output, info_dict = postprocess(output, info_dict)
            output_list.append(output)
        #
        # compute and populate final stats so that it can be used in result
        self.infer_stats_dict = {
            'num_subgraphs': stats_dict['num_subgraphs'],
        }
        if self.settings.target_machine == constants.TARGET_MACHINE_EVM:
            self.infer_stats_dict.update({
                'infer_time_invoke_ms': invoke_time * constants.MILLI_CONST / num_frames,
                'infer_time_core_ms': core_time * constants.MILLI_CONST / num_frames,
                'infer_time_subgraph_ms': subgraph_time * constants.MILLI_CONST / num_frames,
                'ddr_transfer_mb': (ddr_transfer / num_frames_ddr / constants.MEGA_CONST) if num_frames_ddr > 0 else 0
            })
        #
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

    def _import_bev_model(self, description=''):
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
        run_dir_base = os.path.split(session.get_param('run_dir'))[-1]

        is_ok = session.start_import()
        assert is_ok, utils.log_color('\nERROR', f'start_import() did not succeed for:', run_dir_base)

        # Number of temporal frames in BEV detection
        num_bev_temporal_frames = 0
        if 'bev_options:num_temporal_frames' in runtime_options:
            num_bev_temporal_frames = runtime_options['bev_options:num_temporal_frames']

        # Queue for previous feature maps
        if num_bev_temporal_frames > 0:
            self.queue_mem = dict()
            self.queue = queue.Queue(maxsize=num_bev_temporal_frames)

        # for BEVFormer
        # To Do: Use queue for BEVFormer
        prev_bev = None
        for data_index in range(calibration_frames):
            info_dict = {'dataset_info': self.dataset_info, 
                         'label_offset_pred': self.pipeline_config.get('metric',{}).get('label_offset_pred',None),
                         'task_name': self.pipeline_config.get('task_name',{})}
            # Add feature queues to info_dict for preprocessing
            if self.queue is not None:
                info_dict['sample_idx'] = data_index
                info_dict['queue_mem'] = self.queue_mem
                info_dict['queue'] = self.queue
                info_dict['num_bev_temporal_frames'] = num_bev_temporal_frames

            input_data = calibration_dataset[data_index]
            input_data, info_dict = preprocess(input_data, info_dict)

            # For calibration, we cannot add prev_bev from the previous frames.
            # So simply set prev_bev to zero
            # To REVISIT with queue
            if self.pipeline_config.get('task_name', {}) == 'BEVFormer':
                if info_dict['prev_bev_exist'] is False:
                    input_data.append(np.zeros((2500, 1, 256), dtype=np.float32))
                else:
                    input_data.append(prev_bev)

            # this is the actual import
            output, info_dict = session.run_import(input_data, info_dict)
        #

            # For BEVFormer, save output for next frames 
            if self.pipeline_config.get('task_name', {}) == 'BEVFormer':
                prev_bev = output[3]

            # FastBEV: Update queue
            if self.queue is not None:
                if self.queue.full():
                    pop_key = self.queue.get()
                    self.queue_mem.pop(pop_key)

                # add the current feature map
                # it should be batch_size = 1
                self.queue_mem[info_dict['sample_idx']] = \
                    dict(feature_map=output[-1], img_meta=info_dict)
                self.queue.put(info_dict['sample_idx'])

        # close the interpreter
        session.close_interpreter()


    def _infer_bev_frames(self, description=''):
        session = self.pipeline_config['session']
        input_dataset = self.pipeline_config['input_dataset']
        assert input_dataset is not None, f'got input_dataset={input_dataset}. please check settings.dataset_loading'
        preprocess = self.pipeline_config['preprocess']
        postprocess = self.pipeline_config['postprocess']
        runtime_options = session.kwargs['runtime_options']
        run_dir_base = os.path.split(session.get_param('run_dir'))[-1]
        num_frames = self.pipeline_config.get('num_frames', self.settings.num_frames)
        num_frames = min(len(input_dataset), num_frames) if num_frames else len(input_dataset)

        is_ok = session.start_inference()
        assert is_ok, utils.log_color('\nERROR', f'start_infer() did not succeed for:', run_dir_base)

        if self.settings.target_machine == constants.TARGET_MACHINE_EVM:
            invoke_time = 0.0
            core_time = 0.0
            subgraph_time = 0.0
            ddr_transfer = 0.0
            num_frames_ddr = 0

        output_list = []
        pbar_desc = f'infer {description}: {run_dir_base}'

        # Number of temporal frames in BEV detection
        num_bev_temporal_frames = 0
        if 'bev_options:num_temporal_frames' in runtime_options:
            num_bev_temporal_frames = runtime_options['bev_options:num_temporal_frames']

        # Queue for previous feature maps
        if num_bev_temporal_frames > 0:
            self.queue_mem = dict()
            self.queue = queue.Queue(maxsize=num_bev_temporal_frames)

        # for BEVFormer
        # To Do: Use queue for BEVFormer
        prev_bev = None
        for data_index in utils.progress_step(range(num_frames), desc=pbar_desc, position=0):
            info_dict = {'dataset_info': self.dataset_info,
                         'label_offset_pred': self.pipeline_config.get('metric',{}).get('label_offset_pred',None),
                         'task_name': self.pipeline_config.get('task_name',{})}
            # Add feature queues to info_dict for preforce
            if self.queue is not None:
                info_dict['sample_idx'] = data_index
                info_dict['queue_mem'] = self.queue_mem
                info_dict['queue'] = self.queue
                info_dict['num_bev_temporal_frames'] = num_bev_temporal_frames

            data = input_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)

            if self.pipeline_config.get('task_name', {}) == 'BEVFormer':
                if info_dict['prev_bev_exist'] is False:
                    data.append(np.zeros((2500, 1, 256), dtype=np.float32))
                else:
                    data.append(prev_bev)

            # Save input arrays
            #for i in range(len(data)):
            #    data[i].tofile(f"./testdata/bevdet_frame_{data_index:03d}_input_{i}.dat")
            output, info_dict = session.run_inference(data, info_dict)

            # Save output for next frames
            if self.pipeline_config.get('task_name', {}) == 'BEVFormer' or \
               self.pipeline_config.get('task_name', {}) == 'FastBEV_f4':
                prev_bev = output[3]

            # FastBEV: Update queue
            if self.queue is not None:
                if self.queue.full():
                    pop_key = self.queue.get()
                    self.queue_mem.pop(pop_key)

                # add the current feature map
                # it should be batch_size = 1
                self.queue_mem[info_dict['sample_idx']] = \
                    dict(feature_map=output[-1], img_meta=info_dict)
                self.queue.put(info_dict['sample_idx'])

            # Save output arrays
            #for i in range(len(output)):
            #    output[i].tofile(f"./testdata/bevdet_frame_{data_index:03d}_output_{i}.dat")

            stats_dict = session.infer_stats()
            if self.settings.target_machine == constants.TARGET_MACHINE_EVM:
                invoke_time += info_dict['session_invoke_time']
                core_time += stats_dict['core_time']
                subgraph_time += stats_dict['subgraph_time']
                if stats_dict['write_total'] >= 0  and stats_dict['read_total'] >= 0 :
                    ddr_transfer += (stats_dict['write_total'] + stats_dict['read_total'])
                    num_frames_ddr += 1

            if self.settings.flip_test:
                outputs_flip, info_dict = session.run_inference(info_dict['flip_img'], info_dict)
                info_dict['outputs_flip'] = outputs_flip

                stats_dict = session.infer_stats()
                if self.settings.target_machine == constants.TARGET_MACHINE_EVM:
                    invoke_time += info_dict['session_invoke_time']
                    core_time += stats_dict['core_time']
                    subgraph_time += stats_dict['subgraph_time']
                    if stats_dict['write_total'] >= 0  and stats_dict['read_total'] >= 0 :
                        ddr_transfer += (stats_dict['write_total'] + stats_dict['read_total'])
                        num_frames_ddr += 1
            else:
                info_dict['outputs_flip'] = None

            # needed in postprocess to understand the detection threshold set
            info_dict['runtime_options'] = runtime_options

            output, info_dict = postprocess(output, info_dict)
            output_list.append(output)

        #
        # compute and populate final stats so that it can be used in result
        self.infer_stats_dict = {
            'num_subgraphs': stats_dict['num_subgraphs'],
        }
        if self.settings.target_machine == constants.TARGET_MACHINE_EVM:
            self.infer_stats_dict.update({
                'infer_time_invoke_ms': invoke_time * constants.MILLI_CONST / num_frames,
                'infer_time_core_ms': core_time * constants.MILLI_CONST / num_frames,
                'infer_time_subgraph_ms': subgraph_time * constants.MILLI_CONST / num_frames,
                'ddr_transfer_mb': (ddr_transfer / num_frames_ddr / constants.MEGA_CONST) if num_frames_ddr > 0 else 0
            })
        #
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
        metric_options['task_name'] = self.pipeline_config.get('task_name', {})
        metric_options['dataset_category'] = self.pipeline_config['dataset_category']

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
