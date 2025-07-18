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
import re
import wurlitzer
import tarfile

from .. import utils
from .. import datasets
from .model_transformation import *
from .accuracy_pipeline import *
from .gen_config_pipeline import *
from .. import preprocess


class PipelineRunner():
    def __init__(self, settings, pipeline_configs):
        # making a copy of the dataloader for every config can take a lot of time
        # since this happends int he beginning, this can be quite annoying
        # instead it may be better to make a copy opf the whole proc_func just before executaion of it
        self.copy_dataloader = False
        self.settings = settings
        self.pipeline_configs = pipeline_configs
        if settings.sort_pipeline_configs:
            self.pipeline_configs = self._sort_pipeline_configs(self.pipeline_configs)
        #
        self.pipeline_configs = self._filter_pipeline_configs(self.pipeline_configs)

        if settings.model_shortlist is not None or len(self.pipeline_configs) == 0:
            print(utils.log_color('WARNING', f'model_shortlist={settings.model_shortlist}', 'this will cause only a subset of models to be selected for run'))
            print(utils.log_color('WARNING', 'if the model that you wish is not being selected for run, then remove this model_shortlist'))
            print(utils.log_color('WARNING', 'this model_shortlist could be being set in settings_base.yaml or passed inside run_benchmarks_pc.sh'))
        #

    def get_pipeline_configs(self):
        return self.pipeline_configs

    def _sort_pipeline_configs(self, pipeline_configs):
        if pipeline_configs is None:
            return pipeline_configs
        #
        model_shortlist_factors = {model_id:pipeline_config.get('model_info',{}).get('model_shortlist',None) for model_id, pipeline_config in pipeline_configs.items()}
        model_shortlist_factors = dict(sorted(list(model_shortlist_factors.items()), key=lambda d:(d[1] or float('inf'))))
        model_ids_sorted = model_shortlist_factors.keys()
        pipeline_configs_sorted = {model_id:pipeline_configs[model_id] for model_id in model_ids_sorted}
        return pipeline_configs_sorted

    def _filter_pipeline_configs(self, pipeline_configs):
        if pipeline_configs is None:
            return pipeline_configs
        #
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
        pipelines_selected1 = {}
        for model_id, pipeline_config in pipeline_configs.items():
            if self._check_model_selection(self.settings, pipeline_config):
                pipelines_selected1.update({model_id: pipeline_config})
            #
        #

        # additional filtering required
        if self.settings.pipeline_type == constants.PIPELINE_GEN_CONFIG:
            pipelines_selected = {}
            model_path_selected = []
            # we will go in this order of preference
            for supported_session_name in constants.SESSION_NAMES:
                for model_id, pipeline_config in pipelines_selected1.items():
                    model_path = pipeline_config['session'].kwargs['model_path']
                    session_name = pipeline_config['session'].kwargs['session_name']
                    if session_name == supported_session_name:
                        write_gen_config = model_path not in model_path_selected
                        pipeline_config['write_gen_config'] = write_gen_config
                        pipelines_selected.update({model_id: pipeline_config})
                        model_path_selected += [model_path]
                    #
                #
            #
        else:
            pipelines_selected = pipelines_selected1
        #

        if self.settings.config_range is not None:
            pipelines_selected = dict(itertools.islice(pipelines_selected.items(), *self.settings.config_range))
        #
        if self.settings.model_transformation_dict is not None:
            pipelines_selected = model_transformation(self.settings, pipelines_selected)
        #

        # check the datasets and download if they are missing
        pipeline_config_dataset_list = []
        for pipeline_key, pipeline_config in pipelines_selected.items():
            pipeline_config_dataset_list.append(pipeline_config['calibration_dataset'])
            pipeline_config_dataset_list.append(pipeline_config['input_dataset'])
        #
        # sending dataset_list to download_datasets will cause only those to be downloaded and/or loaded
        download_ok = datasets.download_datasets(self.settings, dataset_list=pipeline_config_dataset_list)

        # populate the dataset objects into the pipeline_configs
        if not self.settings.dataset_loading:
            # if dataset_loading is not enabled, that means pipeline_configs are valid, even if dataset is not loaded
            pipelines_final = pipelines_selected
        else:
            # if dataset_loading is set, the dataset should have been loaded at this point for that pipeline_config to be considered
            pipelines_final = {}
            for pipeline_key, pipeline_config in pipelines_selected.items():
                if isinstance(pipeline_config['calibration_dataset'], datasets.DatasetBase) and isinstance(pipeline_config['input_dataset'], datasets.DatasetBase):
                    # if it is an instance of datasets.DatasetBase, we can use it - no need to copy from dataset_cache
                    pipelines_final.update({pipeline_key: pipeline_config})
                elif isinstance(pipeline_config['calibration_dataset'], str) and isinstance(pipeline_config['input_dataset'], str):
                    # if the dataset in pipeline_config is str, we will copy it from dataset_cache
                    calibration_dataset_category = pipeline_config['calibration_dataset']
                    input_dataset_category = pipeline_config['input_dataset']
                    if calibration_dataset_category in self.settings.dataset_cache and input_dataset_category in self.settings.dataset_cache and \
                        isinstance(self.settings.dataset_cache[calibration_dataset_category]['calibration_dataset'], datasets.DatasetBase) and \
                        isinstance(self.settings.dataset_cache[input_dataset_category]['input_dataset'], datasets.DatasetBase):
                        calibration_dataset = self.settings.dataset_cache[calibration_dataset_category]['calibration_dataset']
                        input_dataset = self.settings.dataset_cache[input_dataset_category]['input_dataset']
                        if self.copy_dataloader:
                            print(utils.log_color("\nINFO", f"pipeline_config", f'{pipeline_key} - dataset loader copy for config'))
                            calibration_dataset = copy.deepcopy(calibration_dataset)
                            input_dataset = copy.deepcopy(input_dataset)
                        #
                        pipeline_config['calibration_dataset'] = calibration_dataset
                        pipeline_config['input_dataset'] = input_dataset
                        pipelines_final.update({pipeline_key: pipeline_config})
                    else:
                        print(utils.log_color("\nWARNING", f"pipeline_config", f'{os.path.basename(__file__)}: ignoring the pipeline_config: {pipeline_key}, since the dataset was not loaded: {calibration_dataset_category}, {input_dataset_category}'))
                    #
                else:
                    # if the dataset in pipeline_config is None, we will not bother to copy from dataset_cache, assuming it is not needed.
                    pipelines_final.update({pipeline_key: pipeline_config})
                #
            #
        #
        return pipelines_final

    def _get_log_file(self, pipeline_config):
        if self.settings.log_file:
            log_filename = self.settings.log_file if isinstance(self.settings.log_file, str) else 'run.log'
            log_file = os.path.join(pipeline_config['session'].get_param('run_dir'), log_filename)
        else:
            log_file = None
        #
        return log_file

    def get_tasks(self, separate_import_inference=True, proc_error_regex_list=None):
        # get the cwd so that we can continue even if exception occurs
        cwd = os.getcwd()
        results_list = []
        total = len(self.pipeline_configs)

        proc_error_regex_list = proc_error_regex_list or constants.TIDL_FATAL_ERROR_LOGS_REGEX_LIST
        if self.settings.tensor_bits != 32:
            proc_error_regex_list += constants.TIDL_FATAL_ERROR_LOGS_REGEX_LIST_TENSOR_BITS_NOT_32
        #
        
        task_entries = {}
        for pipeline_index, (model_id, pipeline_config) in enumerate(self.pipeline_configs.items()):
            os.chdir(cwd)
            description = f'{pipeline_index+1}/{total}' if total > 1 else ''
            log_file = self._get_log_file(pipeline_config)
            task_list_for_model = []
            if separate_import_inference:
                # separate import and inference into two tasks - tidl import and inference to run in separate process
                if self.settings.run_import:
                    proc_name = model_id + ':import'
                    basic_settings = self.settings.basic_settings()
                    basic_settings.run_inference = False
                    run_task = functools.partial(self._run_pipeline, basic_settings, pipeline_config, description=description, log_file=log_file)
                    task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_task, 'proc_log':log_file, 'proc_error':proc_error_regex_list})
                #
                if self.settings.run_inference:
                    proc_name = model_id + ':infer'
                    basic_settings = self.settings.basic_settings()
                    basic_settings.run_import = False
                    run_task = functools.partial(self._run_pipeline, basic_settings, pipeline_config, description=description, log_file=log_file)
                    task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_task, 'proc_log':log_file, 'proc_error':proc_error_regex_list})
                #
            else:
                proc_name = model_id
                # note that this basic_settings() copies only the basic settings.
                # sometimes, there is no need to copy the entire settings which includes the dataset_cache
                basic_settings = self.settings.basic_settings()
                run_task = functools.partial(self._run_pipeline, basic_settings, pipeline_config, description=description, log_file=log_file)
                task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_task, 'proc_log':log_file, 'proc_error':proc_error_regex_list})
            #
            task_entries.update({model_id:task_list_for_model})
        #
        return task_entries

    def run(self, task_entries=None):
        if task_entries is None:
            task_entries = self.get_tasks()
        #
        cwd = os.getcwd()
        result_entries = {}
        for task_name, task_list in task_entries.items():
            for proc_entry in task_list:
                os.chdir(cwd)
                proc_func = proc_entry['proc_func']
                # data loader was not copied at initialization - make a copy of the whole proc_func
                if not self.copy_dataloader:
                    proc_func = copy.deepcopy(proc_func)
                #
                result = proc_func()
                result_entries.update({task_name:result})
            #
        #
        return result_entries

    @classmethod
    def _run_pipeline(cls, settings, pipeline_config, description='', log_file=None):
        # proc_errors to check
        proc_errors_regex_list = constants.TIDL_FATAL_ERROR_LOGS_REGEX_LIST
        if settings.tensor_bits != 32:
            proc_errors_regex_list += constants.TIDL_FATAL_ERROR_LOGS_REGEX_LIST_TENSOR_BITS_NOT_32
        #

        # if the run_dir doesn't exist, check if tarfile exists or can be downloaded/untarred
        run_dir = pipeline_config['session'].kwargs['run_dir']
        if not os.path.exists(run_dir):
            work_dir = os.path.dirname(run_dir)
            tarfile_name = run_dir + '.tar.gz'
            if not os.path.exists(tarfile_name):
                tarfile_name = utils.download_file(tarfile_name, work_dir, extract_root=run_dir)
            #
            # extract the tar file
            if (not os.path.exists(run_dir)) and tarfile_name is not None and os.path.exists(tarfile_name):
                os.makedirs(run_dir, exist_ok=True)
                tfp = tarfile.open(tarfile_name)
                tfp.extractall(run_dir)
                tfp.close()
            #
        #

        # capture cwd - to set it later
        cwd = os.getcwd()
        try:
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'a') as log_fp:
                    with wurlitzer.pipes(stdout=log_fp, stderr=wurlitzer.STDOUT):
                        result = cls._run_pipeline_impl(settings, pipeline_config, description)
                    #
                #
                if settings.check_errors:
                    with open(log_file) as log_fp:
                        proc_log_content = log_fp.read()
                        regex_match =  any([re.search(proc_error_entry, proc_log_content) for proc_error_entry in proc_errors_regex_list])
                        if regex_match:
                            result = None
                        #
                    #
                #
            else:
                result = cls._run_pipeline_impl(settings, pipeline_config, description)
            #
        except Exception as e:
            result = None
            model_id = pipeline_config['session'].kwargs['model_id']
            print(utils.log_color('\nERROR', f'model_id:{model_id} run_import:{settings.run_import} run_inference:{settings.run_inference}', str(e)))
            traceback.print_exc()
        #
        # make sure we are in cwd when we return.
        os.chdir(cwd)
        return result

    # this function cannot be an instance method of PipelineRunner, as it causes an
    # error during pickling, involved in the launch of a process is parallel run. make it classmethod
    @classmethod
    def _run_pipeline_impl(cls, settings, pipeline_config, description=''):
        run_dir = pipeline_config['session'].get_param('run_dir')
        print(utils.log_color('\nINFO', 'starting', os.path.basename(run_dir)))
        result = {}
        if settings.pipeline_type == constants.PIPELINE_ACCURACY:
            # use with statement, so that the logger and other file resources are cleaned up
            with AccuracyPipeline(settings, pipeline_config) as accuracy_pipeline:
                result = accuracy_pipeline(description)
            #
        elif settings.pipeline_type == constants.PIPELINE_GEN_CONFIG:
            # this is just an example of how other pipelines can be implemented.
            # 'something' used here is not real and it is not supported
            with GenConfigPipeline(settings, pipeline_config) as gen_config_pipeline:
                result = gen_config_pipeline(description)
            #
        else:
            assert False, f'ERROR: unknown pipeline: {settings.pipeline_type}'
        #
        return result

    def _str_match_any(self, k, x_list):
        match_any = any([(k in x) for x in x_list])
        return match_any

    def _str_match_plus(self, ks, x_list):
        ks = ks.split('+')
        match_fully = all([self._str_match_any(k,x_list) for k in ks])
        return match_fully

    def _str_match_plus_any(self, keywords, search_list):
        any_match_fully = any([self._str_match_plus(kw, search_list) if kw else True for kw in keywords])
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
        if settings.model_shortlist is not None:
            if isinstance(settings.model_shortlist, (int,)):
                model_shortlist = pipeline_config['model_info'].get('model_shortlist', None)
                selected_model = selected_model and (model_shortlist is not None and model_shortlist <= settings.model_shortlist)
            else:
                assert False, f"ERROR: invalid value or type for model_shortlist (type:int): {settings.model_shortlist}, (type:{type(settings.model_shortlist)})"
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
