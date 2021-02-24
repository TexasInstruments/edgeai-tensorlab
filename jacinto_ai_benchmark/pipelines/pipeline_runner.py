import functools
import itertools
from .. import constants
from .accuracy_pipeline import *
from .. import utils


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
        self.pipeline_configs = pipelines_selected

    def run(self):
        if self.settings.parallel_devices is not None:
            return self._run_pipelines_parallel()
        else:
            return self._run_pipelines_sequential()
        #

    def _run_pipelines_sequential(self):
        # get the cwd so that we can continue even if exception occurs
        cwd = os.getcwd()
        results_list = []
        total = len(self.pipeline_configs)
        for pipeline_id, pipeline_config in enumerate(self.pipeline_configs.values()):
            os.chdir(cwd)
            description = f'{pipeline_id+1}/{total}'
            result = self._run_pipeline(pipeline_config, description=description,
                                        enable_logging=self.settings.enable_logging)
            results_list.append(result)
        #
        return results_list

    def _run_pipelines_parallel(self):
        # get the cwd so that we can continue even if exception occurs
        cwd = os.getcwd()
        num_devices = len(self.settings.parallel_devices) if self.settings.parallel_devices is not None else 0
        description = ' '*120 + 'TASKS'
        parallel_exec = utils.ParallelRun(num_processes=num_devices, desc=description)
        for pipeline_index, pipeline_config in enumerate(self.pipeline_configs.values()):
            os.chdir(cwd)
            parallel_device = self.settings.parallel_devices[pipeline_index % num_devices] \
                if self.settings.parallel_devices is not None else 0
            run_pipeline_bound_func = functools.partial(self._run_pipeline, pipeline_config,
                                                        parallel_device=parallel_device, description='',
                                                        enable_logging=self.settings.enable_logging)
            parallel_exec.enqueue(run_pipeline_bound_func)
        #
        results_list = parallel_exec.run()
        return results_list

    # this function cannot be an instance method of PipelineRunner, as it causes an
    # error during pickling, involved in the launch of a process is parallel run. make it classmethod
    @classmethod
    def _run_pipeline(cls, pipeline_config, parallel_device=None, description='', enable_logging=True):
        if parallel_device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)
        #
        result = {}
        pipeline_types = utils.as_list(pipeline_config['pipeline_type'])
        try:
            for pipeline_type in pipeline_types:
                if pipeline_type == constants.PIPELINE_ACCURACY:
                    # use with statement, so that the logger and other file resources are cleaned up
                    with AccuracyPipeline(pipeline_config, enable_logging=enable_logging) as accuracy_pipeline:
                        accuracy_result = accuracy_pipeline.run(description)
                        result.update(accuracy_result)
                    #
                elif pipeline_type == constants.PIPELINE_SOMETHING:
                    # this is just an example of how other pipelines can be implemented.
                    # 'something' used here is not real and it is not supported
                    with SomethingPipeline(pipeline_config) as something_pipeline:
                        something_result = something_pipeline.run()
                        result.update(something_result)
                    #
                else:
                    assert False, f'unknown pipeline: {pipeline_type}'
                #
        except Exception as e:
            print(f'\n{str(e)}')
        #
        return result

    def _check_model_selection(self, config, pipeline_config):
        if config.model_selection is not None:
            selected_model = False
            model_path = pipeline_config['session'].get_param('model_path')
            model_id = pipeline_config['session'].get_param('model_id')
            model_selection = utils.as_list(config.model_selection)
            for keyword in model_selection:
                if keyword in model_path:
                    selected_model = True
                #
                if keyword in model_id:
                    selected_model = True
                #
            #
        else:
            selected_model = True
        #
        return selected_model

