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
import copy
import yaml

import edgeai_benchmark
from .... import utils
from .. import constants


this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '../../../../../'))

edgeai_benchmark_path = os.path.join(repo_parent_path, 'edgeai-benchmark')

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
        self.artifact_ext = '.tar.gz'
        self.result = None

        # prepare for model compilation
        self._prepare_pipeline_config()

        if self.params.compilation.tidl_offload:
            progress_regex = \
                {'type':'Progress', 'name':'Progress', 'description':'Progress of Compilation', 'unit':'Iteration', 'value':None,
                  'regex':[{'op':'search', 'pattern':r'Calibration iteration number\s+(?<Progress>\d+)\s+completed', 'group':1}],
                }
        else:
            progress_regex = \
                {'type':'Progress', 'name':'Progress', 'description':'Progress of Compilation', 'unit':'Frame', 'value':None,
                 'regex':[{'op':'search', 'pattern':r'infer\s+\:\s+.*?\s+(?<infer>\d+)', 'group':1}],
                 }
        #

        if self.params.common.task_type == constants.TASK_TYPE_CLASSIFICATION:
            log_summary_regex = {
                'js': [
                    progress_regex,
                    {'type':'Validation Accuracy', 'name':'Accuracy', 'description':'Accuracy of Compilation', 'unit':'Accuracy Top-1%', 'value':None,
                     'regex':[{'op':'search', 'pattern':r'benchmark results.*?accuracy_top1.*?\:\s+(?<accuracy>\d+\.\d+)', 'group':1, 'dtype':'float', 'scale_factor':1}],
                     },
                    {'type':'Completed', 'name':'Completed', 'description':'Completion of Compilation', 'unit':None, 'value':None,
                     'regex':[{'op':'search', 'pattern':r'success\:.*compilation\s+completed', 'group':1, 'dtype':'str', 'case_sensitive':False}],
                     },
                ]
            }
        elif self.params.common.task_type == constants.TASK_TYPE_DETECTION:
            log_summary_regex = {
                'js': [
                    progress_regex,
                    {'type':'Validation Accuracy', 'name':'Accuracy', 'description':'Accuracy of Compilation', 'unit':'AP50%', 'value':None,
                     'regex':[{'op':'search', 'pattern':r'benchmark results.*?accuracy_ap50\%.*?\:\s+(?<accuracy>\d+\.\d+)', 'group':1, 'dtype':'float', 'case_sensitive':False, 'scale_factor':1}],
                     },
                    {'type':'Completed', 'name':'Completed', 'description':'Completion of Compilation', 'unit':None, 'value':None,
                     'regex':[{'op':'search', 'pattern':r'success\:.*compilation\s+completed', 'group':1, 'dtype':'str', 'case_sensitive':False}],
                     },
                ]
            }
        elif self.params.common.task_type == constants.TASK_TYPE_SEGMENTATION:
            # TODO: this needs to be corrected for segmentation
            log_summary_regex = {
                'js': [
                    progress_regex,
                    {'type':'Validation Accuracy', 'name':'Accuracy', 'description':'Accuracy of Compilation', 'unit':'MeanIoU%', 'value':None,
                     'regex':[{'op':'search', 'pattern':r'benchmark results.*?accuracy_mean_iou\%.*?\:\s+(?<accuracy>\d+\.\d+)', 'group':1, 'dtype':'float', 'case_sensitive':False, 'scale_factor':1}],
                     },
                    {'type':'Completed', 'name':'Completed', 'description':'Completion of Compilation', 'unit':None, 'value':None,
                     'regex':[{'op':'search', 'pattern':r'success\:.*compilation\s+completed', 'group':1, 'dtype':'str', 'case_sensitive':False}],
                     },
                ]
            }
        elif self.params.common.task_type == constants.TASK_TYPE_KEYPOINT_DETECTION:
            # TODO: this needs to be corrected for keypoint detection
            log_summary_regex = {
                'js': [
                    progress_regex,
                    {'type':'Validation Accuracy', 'name':'Accuracy', 'description':'Accuracy of Compilation', 'unit':'AP50%', 'value':None,
                     'regex':[{'op':'search', 'pattern':r'benchmark results.*?accuracy_ap50\%.*?\:\s+(?<accuracy>\d+\.\d+)', 'group':1, 'dtype':'float', 'case_sensitive':False, 'scale_factor':1}],
                     },
                    {'type':'Completed', 'name':'Completed', 'description':'Completion of Compilation', 'unit':None, 'value':None,
                     'regex':[{'op':'search', 'pattern':r'success\:.*compilation\s+completed', 'group':1, 'dtype':'str', 'case_sensitive':False}],
                     },
                ]
            }
        else:
            log_summary_regex = None
        #

        model_compiled_path = self.run_dir
        packaged_artifact_path = self._get_packaged_artifact_path() # actual, internal, short path
        model_packaged_path = self._get_final_artifact_path() # a more descriptive symlink

        self.params.update(
            compilation=utils.ConfigDict(
                model_compiled_path=model_compiled_path,
                log_file_path=os.path.join(model_compiled_path, 'run.log'),
                log_summary_regex=log_summary_regex,
                summary_file_path=os.path.join(model_compiled_path, 'summary.yaml'),
                output_tensors_path=os.path.join(model_compiled_path, 'outputs'),
                model_packaged_path=model_packaged_path, # final compiled package
                model_visualization_path=os.path.join(model_compiled_path, 'artifacts', 'tempDir', 'runtimes_visualization.svg'),
            )
        )

    def clear(self):
        # clear the dirs
        shutil.rmtree(self.params.compilation.compilation_path, ignore_errors=True)

    def run(self):
        ''''
        The actual compilation function.
        '''
        self.result = None
        self.result = edgeai_benchmark.interfaces.run_benchmark_config(self.settings, self.work_dir, self.pipeline_configs)

        # remove special characters
        utils.cleanup_special_chars(self.params.compilation.log_file_path)

        # package artifacts
        edgeai_benchmark.interfaces.package_artifacts(self.settings, self.work_dir, out_dir=self.package_dir, custom_model=True)
        
        # make a symlink to the packaged artifacts
        # internally we use a short path as tidl has file path length restrictions
        # for use outside, create symlink to a more descriptive file
        packaged_artifact_path = self._get_packaged_artifact_path()
        utils.make_symlink(packaged_artifact_path, self.params.compilation.model_packaged_path)
        return self.params

    def get_result(self):
        return self.result

    def _get_benchmark_settings(self, model_selection, calib_dataset, val_dataset):
        settings_file = edgeai_benchmark.get_settings_file(target_machine=self.params.common.target_machine, with_model_import=True)

        # do not let the calibration_frames exceed calibration_dataset size
        calibration_frames = min(self.params.compilation.calibration_frames, len(calib_dataset))

        settings = edgeai_benchmark.config_settings.ConfigSettings(
                        settings_file,
                        target_device=self.params.common.target_device,
                        model_selection=model_selection,
                        modelartifacts_path=self.params.compilation.compilation_path,
                        tensor_bits=self.params.compilation.tensor_bits,
                        calibration_frames=calibration_frames,
                        calibration_iterations=self.params.compilation.calibration_iterations,
                        num_frames=self.params.compilation.num_frames,
                        runtime_options=None,
                        detection_threshold=self.params.compilation.detection_threshold,
                        detection_top_k=self.params.compilation.detection_top_k,
                        parallel_devices=None,   # 0 or None will use the first cuda/gpu if tidl_tools with gpu support is used
                        parallel_processes=1,    # do the compilation in a new processes for more stability
                        dataset_loading=False,
                        save_output=self.params.compilation.save_output,
                        input_optimization=self.params.compilation.input_optimization,
                        tidl_offload=self.params.compilation.tidl_offload,
                        num_output_frames=self.params.compilation.num_output_frames,
                        log_file=self.params.compilation.log_file
        )
        return settings

    def _prepare_pipeline_config(self):
        '''
        prepare for model compilation
        '''
        dataset_kwargs = {}
        if self.params.common.task_type == 'detection':
            dataset_loader = edgeai_benchmark.datasets.ModelMakerDetectionDataset
        elif self.params.common.task_type == 'classification':
            dataset_loader = edgeai_benchmark.datasets.ModelMakerClassificationDataset
        elif self.params.common.task_type == 'segmentation':
            dataset_loader = edgeai_benchmark.datasets.ModelMakerSegmentationDataset
            dataset_kwargs.update(dict(with_background_class=self.params.training.with_background_class))
        elif self.params.common.task_type == 'keypoint_detection':
            dataset_loader = edgeai_benchmark.datasets.ModelMakerKeypointDetectionDataset
        else:
            dataset_loader = None
        #

        # can use any suitable data loader provided in datasets folder of edgeai-benchmark or write another
        calib_dataset = dataset_loader(
            path=self.params.dataset.dataset_path,
            split=self.params.dataset.split_names[0],
            shuffle=True,
            num_frames=self.params.compilation.calibration_frames, # num_frames is not critical here,
            annotation_prefix=self.params.dataset.annotation_prefix,
            **dataset_kwargs
        )
        calib_dataset.kwargs['type'] = dataset_loader.__name__ # insert type to be able to reconstruct it back from yaml file
        val_dataset = dataset_loader(
            path=self.params.dataset.dataset_path,
            split=self.params.dataset.split_names[1],
            shuffle=False, # can be set to True as well, if needed
            num_frames=self.params.compilation.num_frames, # this num_frames is important for accuracy calculation
            annotation_prefix=self.params.dataset.annotation_prefix,
            **dataset_kwargs
        )
        val_dataset.kwargs['type'] = dataset_loader.__name__ # insert type to be able to reconstruct it back from yaml file

        self.settings = self._get_benchmark_settings(self.params.compilation.model_compilation_id, calib_dataset, val_dataset)
        self.work_dir, self.package_dir = self._get_base_dirs()
        self.run_dir = self._get_run_dir()

        # it may be easier to get the existing config and modify the aspects that need to be changed
        if hasattr(calib_dataset, "num_keypoints"):
            self.settings['num_kpts'] = calib_dataset.num_keypoints
        #
        pipeline_configs = edgeai_benchmark.interfaces.select_configs(self.settings, self.work_dir, adjust_config=False)
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

        # use a short path for the compiled artifacts dir
        pipeline_config['session'].set_param('run_dir', self.run_dir)

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
        # num_classes may not have been available during dataset instanciation and hence color_map might not have ben created
        # update it here.
        dataset_info = val_dataset.get_dataset_info()
        if 'color_map' not in dataset_info or dataset_info['color_map'] is None:
            num_classes = val_dataset.get_num_classes()
            dataset_info['color_map'] = val_dataset.get_color_map(num_classes)
        #
        if isinstance(pipeline_config['metric'], dict) and 'label_offset_pred' in pipeline_config['metric']:
            dataset_info = val_dataset.get_dataset_info()
            categories = dataset_info['categories']
            min_cat_id = min([cat['id'] for cat in categories])
            pipeline_config['metric']['label_offset_pred'] = min_cat_id
        #
        postprocess = pipeline_config['postprocess']
        for transform in postprocess.transforms:
            # special case
            if hasattr(transform, 'update_color_map'):
                dataset_info = val_dataset.get_dataset_info()
                transform.update_color_map(dataset_info['color_map'])
            #
        #
        self.pipeline_configs = pipeline_configs

    # compiled_artifact and packaged artifact uses a short name using model_compilation_id
    # as tidl has limitations in path length
    def _get_run_dir(self):
        run_dir = os.path.join(self.work_dir, self.params.compilation.model_compilation_id)
        return run_dir

    # compiled_artifact and packaged artifact uses a short name using model_compilation_id
    # as tidl has limitations in path length
    def _get_packaged_artifact_path(self):
        packaged_artifact_path = os.path.join(self.package_dir, self.params.compilation.model_compilation_id) + self.artifact_ext
        return packaged_artifact_path

    # final_artifact_name is a more descriptive name with the actual name of the model
    # this will be used to create a symlink to the packaged_artifact_path
    def _get_final_artifact_path(self):
        pipeline_config = list(self.pipeline_configs.values())[0]
        session_name = pipeline_config['session'].get_param('session_name')
        target_device_suffix = self.params.common.target_device
        run_name_splits = list(os.path.split(self.params.common.run_name))
        final_artifact_name = '_'.join(run_name_splits + [session_name, target_device_suffix])
        final_artifact_path = os.path.join(self.package_dir, final_artifact_name) + self.artifact_ext
        return final_artifact_path

    def _has_logs(self):
        log_dir = self.run_dir
        if (log_dir is None) or (not os.path.exists(log_dir)):
            return False
        #
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if len(log_files) == 0:
            return False
        #
        return True

    def _get_base_dirs(self):
        work_dir = os.path.join(self.settings.modelartifacts_path, 'work')
        package_dir = os.path.join(self.settings.modelartifacts_path, 'pkg')
        return work_dir, package_dir

    def get_params(self):
        return self.params
