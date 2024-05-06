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

import sys
import functools
import itertools
import warnings
import copy
import traceback

from .. import utils
from .. import datasets
from .model_transformation import *
from .accuracy_pipeline import *
from edgeai_benchmark import preprocess


class PipelineRunner():
    def __init__(self, settings, pipeline_configs):
        self.settings = settings
        for model_id, pipeline_config in pipeline_configs.items():
            # set model_id in each config
            pipeline_config['session'].set_param('model_id', model_id)
            session = pipeline_config['session']
            # call initialize() on each pipeline_config so that run_dir,
            # artifacts folder and such things are initialized
            session.initialize()
            # get the meta params if it is present and populate model_info - this is just for information
            # commenting out, as this add additional python package dependency
            # od_meta_names_key = 'object_detection:meta_layers_names_list'
            # runtime_options = session.get_param('runtime_options')
            # meta_path = runtime_options.get(od_meta_names_key, None)
            # if meta_path is not None and isinstance(meta_path, str) and os.path.splitext(meta_path)[-1] == '.prototxt':
            #     meta_info = self._parse_prototxt(meta_path)
            #     model_info = pipeline_config.get('model_info', {})
            #     model_info[od_meta_names_key] = meta_info
            # #
        #
        # short list a set of models based on the wild card given in model_selection
        pipelines_selected = {}
        for model_id, pipeline_config in pipeline_configs.items():
            if self._check_model_selection(self.settings, pipeline_config):
                pipelines_selected.update({model_id: pipeline_config})
            #
        #
        if settings.config_range is not None:
            pipelines_selected = dict(itertools.islice(pipelines_selected.items(), *settings.config_range))
        #
        if settings.model_transformation_dict is not None:
            pipelines_selected = model_transformation(settings, pipelines_selected)
        #
        self.pipeline_configs = pipelines_selected

        # check the datasets and download if they are missing
        pipeline_config_dataset_list = []
        for pipeline_key, pipeline_config in self.pipeline_configs.items():
            pipeline_config_dataset_list.append(pipeline_config['calibration_dataset'])
            pipeline_config_dataset_list.append(pipeline_config['input_dataset'])
        #
        # sending dataset_list to download_datasets will cause only those to be downloaded
        download_ok = datasets.download_datasets(settings, dataset_list=pipeline_config_dataset_list)
        # populate the dataset objects into the pipeline_configs
        for pipeline_key, pipeline_config in self.pipeline_configs.items():
            if isinstance(pipeline_config['calibration_dataset'], str):
                dataset_category_name = pipeline_config['calibration_dataset']
                pipeline_config['calibration_dataset'] = copy.deepcopy(settings.dataset_cache[dataset_category_name]['calibration_dataset'])
            #
            if isinstance(pipeline_config['input_dataset'], str):
                dataset_category_name = pipeline_config['input_dataset']
                pipeline_config['input_dataset'] = copy.deepcopy(settings.dataset_cache[dataset_category_name]['input_dataset'])
            #
        #

    def run(self):
        if self.settings.parallel_processes in (None, 0):
            return self._run_pipelines_sequential()
        else:
            return self._run_pipelines_parallel()
        #

    def _run_pipelines_sequential(self):
        # get the cwd so that we can continue even if exception occurs
        cwd = os.getcwd()
        results_list = []
        total = len(self.pipeline_configs)
        for pipeline_id, pipeline_config in enumerate(self.pipeline_configs.values()):
            os.chdir(cwd)
            description = f'{pipeline_id+1}/{total}' if total > 1 else ''
            result = self._run_pipeline(self.settings, pipeline_config, description=description)
            results_list.append(result)
        #
        return results_list

    def _run_pipelines_parallel(self):
        if self.settings.parallel_devices in (None, 0):
            parallel_devices = [0]
        else:
            parallel_devices = range(self.settings.parallel_devices) if isinstance(self.settings.parallel_devices, int) \
                else self.settings.parallel_devices
        #
        cwd = os.getcwd()
        description = 'TASKS'
        parallel_exec = utils.ParallelRun(parallel_processes=self.settings.parallel_processes, parallel_devices=parallel_devices,
                                          desc=description)
        for pipeline_index, pipeline_config in enumerate(self.pipeline_configs.values()):
            os.chdir(cwd)
            run_pipeline_bound_func = functools.partial(self._run_pipeline, self.settings, pipeline_config,
                                                        description='')
            parallel_exec.enqueue(run_pipeline_bound_func)
        #
        results_list = parallel_exec.run()
        return results_list

    # this function cannot be an instance method of PipelineRunner, as it causes an
    # error during pickling, involved in the launch of a process is parallel run. make it classmethod
    @classmethod
    def _run_pipeline_impl(cls, settings, pipeline_config, description=''):
        result = {}
        if settings.pipeline_type == constants.PIPELINE_ACCURACY:
            # use with statement, so that the logger and other file resources are cleaned up
            with AccuracyPipeline(settings, pipeline_config) as accuracy_pipeline:
                accuracy_result = accuracy_pipeline(description)
                result.update(accuracy_result)
            #
        elif settings.pipeline_type == constants.PIPELINE_SOMETHING:
            # this is just an example of how other pipelines can be implemented.
            # 'something' used here is not real and it is not supported
            with SomethingPipeline(settings, pipeline_config) as something_pipeline:
                something_result = something_pipeline(description)
                result.update(something_result)
            #
        else:
            assert False, f'unknown pipeline: {settings.pipeline_type}'
        #
        return result

    @classmethod
    def _run_pipeline(cls, settings, pipeline_config, description=''):
        # note that this basic_settings() copies only the basic settings.
        # sometimes, there is no need to copy the entire settings which includes the dataset_cache
        basic_settings = settings.basic_settings()

        # capture cwd - to set it later
        cwd = os.getcwd()
        result = {}
        try:
            run_dir = pipeline_config['session'].get_param('run_dir')
            print(utils.log_color('\nINFO', 'starting', os.path.basename(run_dir)))
            result = cls._run_pipeline_impl(basic_settings, pipeline_config, description)
        except Exception as e:
            traceback.print_exc()
            print(str(e))
        #

        # make sure we are in cwd when we return.
        os.chdir(cwd)
        return result

    def _str_match_any(self, k, x_list):
        match_any = any([(k in x) for x in x_list])
        return match_any

    def _str_match_plus(self, ks, x_list):
        ks = ks.split('+')
        match_fully = all([self._str_match_any(k,x_list) for k in ks])
        return match_fully

    def _str_match_plus_any(self, keywords, search_list):
        any_match_fully = any([self._str_match_plus(kw, search_list) for kw in keywords])
        return any_match_fully

    def _check_model_selection(self, settings, pipeline_config):
        model_path = pipeline_config['session'].get_param('model_path')
        model_id = pipeline_config['session'].get_param('model_id')
        model_path0 = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
        model_type = pipeline_config['session'].get_param('model_type')
        model_type = model_type or os.path.splitext(model_path0)[1][1:]
        selected_model = True
        if settings.runtime_selection is not None:
            runtime_selection = utils.as_list(settings.runtime_selection)
            if pipeline_config['session'].get_session_name() not in runtime_selection:
                selected_model = False
            #
        #
        if settings.task_selection is not None:
            task_selection = utils.as_list(settings.task_selection)
            if pipeline_config['task_type'] not in task_selection:
                selected_model = False
            #
        #
        if isinstance(settings.model_shortlist, (int,float)):
            if settings.model_shortlist is not None:
                model_shortlist = pipeline_config['model_info'].get('model_shortlist', None)
                selected_model = selected_model and (model_shortlist is not None and model_shortlist <= settings.model_shortlist)
            #
        #
        if settings.model_selection is not None:
            model_selection = utils.as_list(settings.model_selection)
            selected_model = selected_model and self._str_match_plus_any(model_selection, (model_path0,model_id,model_type))
        #
        if settings.model_exclusion is not None:
            model_exclusion = utils.as_list(settings.model_exclusion)
            excluded_model = self._str_match_plus_any(model_exclusion, (model_path0,model_id,model_type))
            selected_model = selected_model and (not excluded_model)
        #
        if settings.dataset_selection is not None:
            dataset_selection = utils.as_list(settings.dataset_selection)
            dataset_category = pipeline_config.get('dataset_category', None)
            assert dataset_category is not None, f'dataset_selection is set, but dataset_category is not defined in pipeline_config: {pipeline_config}'
            selected_model = selected_model and (dataset_category in dataset_selection)
        #
        return selected_model
