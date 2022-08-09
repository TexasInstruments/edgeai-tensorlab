#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import os
import shutil
import jai_benchmark
from .... import utils


class ModelCompilation():
    @classmethod
    def init_params(self, *args, **kwargs):
        params = dict(
            compilation=dict(
            )
        )
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event
        self.settings_file = jai_benchmark.get_settings_file(target_machine=self.params.common.target_machine, with_model_import=True)
        self.settings = self._get_settings(model_selection=self.params.compilation.model_compilation_id)
        # prepare for model compilation
        self._prepare()
        # update params that are specific to this backend and model
        model_compiled_path = self._get_log_dir()
        model_packaged_path = self._get_output_file()
        self.params.update(
            compilation=utils.ConfigDict(
                log_file_path=os.path.join(model_compiled_path, 'run.log'),
                model_compiled_path=model_compiled_path,
                model_packaged_path=model_packaged_path,
            )
        )

    def clear(self):
        # clear the dirs
        shutil.rmtree(self.params.compilation.compilation_path, ignore_errors=True)

    def _prepare(self):
        '''
        prepare for model compilation
        '''
        self.work_dir, self.package_dir = self._get_base_dirs()

        if self.params.common.task_type == 'detection':
            dataset_loader = jai_benchmark.datasets.ModelMakerDetectionDataset
        elif self.params.common.task_type == 'classification':
            dataset_loader = jai_benchmark.datasets.ModelMakerClassificationDataset
        else:
            dataset_loader = None
        #

        # can use any suitable data loader provided in datasets folder of edgeai-benchmark or write another
        calib_dataset = dataset_loader(
            path=self.params.dataset.dataset_path,
            split='train',
            shuffle=True,
            num_frames=self.params.compilation.calibration_frames, # num_frames is not critical here,
            annotation_prefix=self.params.dataset.annotation_prefix
        )
        val_dataset = dataset_loader(
            path=self.params.dataset.dataset_path,
            split='val',
            shuffle=False, # can be set to True as well, if needed
            num_frames=self.params.compilation.num_frames, # this num_frames is important for accuracy calculation
            annotation_prefix=self.params.dataset.annotation_prefix
        )

        # it may be easier to get the existing config and modify the aspects that need to be changed
        pipeline_configs = jai_benchmark.tools.select_configs(self.settings, self.work_dir)
        num_pipeline_configs = len(pipeline_configs)
        assert num_pipeline_configs == 1, f'specify a unique model name in edgeai-benchmark. found {num_pipeline_configs} configs'
        pipeline_config = list(pipeline_configs.values())[0]

        # dataset settings
        pipeline_config['calibration_dataset'] = calib_dataset
        pipeline_config['input_dataset'] = val_dataset

        # preprocess
        preprocess = pipeline_config['preprocess']
        preprocess.set_input_size(resize=self.params.training.input_resize, crop=self.params.training.input_cropsize)

        # session
        pipeline_config['session'].set_param('work_dir', self.work_dir)
        pipeline_config['session'].set_param('target_device', self.settings.target_device)
        pipeline_config['session'].set_param('model_path', self.params.training.model_export_path)
        # reset - will change based on the model_path given here
        pipeline_config['session'].set_param('run_dir', None)

        runtime_options = pipeline_config['session'].get_param('runtime_options')
        self.meta_layers_names_list = 'object_detection:meta_layers_names_list'
        if self.meta_layers_names_list in runtime_options:
            runtime_options[self.meta_layers_names_list] = self.params.training.model_proto_path
        #
        runtime_options.update(self.params.compilation.get('runtime_options', {}))

        # model_info:metric_reference defined in benchmark code is for the pretrained model - remove it.
        metric_reference = pipeline_config['model_info']['metric_reference']
        for k, v in metric_reference.items():
            metric_reference[k] = None # TODO: get from training
        #

        # metric
        if 'metric' in pipeline_config:
            pipeline_config['metric'].update(self.params.compilation.get('metric', {}))
        elif 'metric' in self.params.compilation:
            pipeline_config['metric'] = self.params.compilation.metric
        #
        if isinstance(pipeline_config['metric'], dict) and 'label_offset_pred' in pipeline_config['metric']:
            dataset_info = val_dataset.get_dataset_info()
            categories = dataset_info['categories']
            min_cat_id = min([cat['id'] for cat in categories])
            pipeline_config['metric']['label_offset_pred'] = min_cat_id
        #
        self.pipeline_configs = pipeline_configs

    def run(self):
        ''''
        The actual compilation function. Move this to a worker process, if this function is called from a GUI.
        '''
        # modify the confidence threshold for detection, if required
        pipeline_config = list(self.pipeline_configs.values())[0]
        runtime_options = pipeline_config['session'].get_param('runtime_options')
        if self.meta_layers_names_list in runtime_options:
            self._replace_confidence_threshold(self.params.training.model_proto_path)
            runtime_options[self.meta_layers_names_list] = self.params.training.model_proto_path
        #
        # run the accuracy pipeline
        jai_benchmark.tools.run_accuracy(self.settings, self.work_dir, self.pipeline_configs)
        # package artifacts
        jai_benchmark.tools.package_artifacts(self.settings, self.work_dir, out_dir=self.package_dir)
        return self.params

    def _get_settings(self, model_selection=None):
        runtime_options = dict(accuracy_level=self.params.compilation.accuracy_level)
        target_device_suffix = self.params.common.target_device.lower()
        settings = jai_benchmark.config_settings.ConfigSettings(
                        self.settings_file,
                        target_device=self.params.common.target_device,
                        run_suffix=target_device_suffix,
                        model_selection=model_selection,
                        modelartifacts_path=self.params.compilation.compilation_path,
                        tensor_bits=self.params.compilation.tensor_bits,
                        calibration_frames=self.params.compilation.calibration_frames,
                        calibration_iterations=self.params.compilation.calibration_iterations,
                        num_frames=self.params.compilation.num_frames,
                        runtime_options=runtime_options,
                        detection_thr=self.params.compilation.detection_thr,
                        parallel_devices=None,
                        dataset_loading=False,
                        save_output=self.params.compilation.save_output)
        return settings

    def _get_log_dir(self,):
        pipeline_config = list(self.pipeline_configs.values())[0]
        run_dir = pipeline_config['session'].get_run_dir()
        return run_dir

    def _get_output_file(self):
        work_dir, package_dir = self._get_base_dirs()
        run_dir = self._get_log_dir()
        compiled_package_file = run_dir.replace(work_dir, package_dir) + '.tar.gz'
        return compiled_package_file

    def _has_logs(self):
        log_dir = self._get_log_dir()
        if (log_dir is None) or (not os.path.exists(log_dir)):
            return False
        #
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if len(log_files) == 0:
            return False
        #
        return True

    def _get_base_dirs(self):
        work_dir = os.path.join(self.settings.modelartifacts_path, 'modelartifacts', f'{self.settings.tensor_bits}bits')
        package_dir = os.path.join(f'{self.settings.modelartifacts_path}', 'modelartifacts_package', f'{self.settings.tensor_bits}bits')
        return work_dir, package_dir

    def _replace_confidence_threshold(self, filename):
        if not filename.endswith('.prototxt'):
            return
        #
        space_string = ' '
        with open(filename) as fp:
            lines = fp.readlines()
        #
        for line_index, line in enumerate(lines):
            line = line.rstrip()
            match_str1 = 'confidence_threshold:'
            replacement_key1 = line.lstrip(space_string).split(space_string)[0]
            if match_str1 == replacement_key1: # exact match
                replacement_str1 = f'{replacement_key1} {self.settings.detection_thr}'
                leading_spaces = len(line) - len(line.lstrip(space_string))
                line = space_string * leading_spaces + replacement_str1
            #
            if self.settings.detection_thr < 0.3:
                match_str2 = 'top_k:'
                replacement_key2 = line.lstrip(space_string).split(space_string)[0]
                if match_str2 == replacement_key2: # exact match
                    replacement_str2 = f'{replacement_key2} 500'
                    leading_spaces = len(line) - len(line.lstrip(space_string))
                    line = space_string * leading_spaces + replacement_str2
                #
            #
            lines[line_index] = line
        #
        with open(filename, 'w') as fp:
            fp.write('\n'.join(lines))
        #

    def get_params(self):
        return self.params
