import os
import atpbar
from .. import utils

class AccuracyPipeline():
    def __init__(self):
        self.info_dict = dict()

    def run(self, pipeline_config):
        result = {}
        session = pipeline_config['session']
        run_import = pipeline_config['run_import']
        run_inference = pipeline_config['run_inference']

        if run_import:
            self.import_model(session, pipeline_config)
        #
        if run_inference:
            output_list = self.infer_frames(session, pipeline_config)
            result = self.evaluate(session, pipeline_config, output_list)
        #
        return result

    def import_model(self, session, pipeline_config):
        calibration_dataset = pipeline_config['calibration_dataset']
        preprocess = pipeline_config['preprocess']
        description = os.path.split(session.get_work_dir())[-1]
        if pipeline_config['verbose_mode']:
            print('import & calibration: ' + description)
        #
        calib_data = []
        num_frames = len(calibration_dataset)
        for data_index in range(num_frames):
            data = calibration_dataset[data_index]
            data = self._sequential_pipeline(preprocess, data)
            calib_data.append(data)
        #

        session.import_model(calib_data)

    def infer_frames(self, session, pipeline_config):
        input_dataset = pipeline_config['input_dataset']
        preprocess = pipeline_config['preprocess']
        postprocess = pipeline_config['postprocess']
        description = os.path.split(session.get_work_dir())[-1]

        is_ok = session.start_infer()
        assert is_ok, f'start_infer() did not succeed for {description}'

        output_list = []
        num_frames = len(input_dataset)
        for data_index in atpbar.atpbar(range(num_frames), name='inference: ' + description):
            data = input_dataset[data_index]
            data = self._sequential_pipeline(preprocess, data)
            output = self._run_session(session, data)
            output = self._sequential_pipeline(postprocess, output)
            output_list.append(output)
        #
        return output_list

    def evaluate(self, session, pipeline_config, output_list):
        # if metric is not given use input_dataset
        if 'metric' in pipeline_config and callable(pipeline_config['metric']):
            metric = pipeline_config['metric']
            metric_options = {}
        else:
            metric = pipeline_config['input_dataset']
            metric_options = pipeline_config.get('metric', {})
        #
        work_dir = session.get_work_dir()
        metric_options['work_dir'] = work_dir
        metric = utils.as_list(metric)
        metric_options = utils.as_list(metric_options)
        output_dict = {}
        for m, m_options in zip(metric, metric_options):
            output = m(output_list, **m_options)
            output_dict.update(output)
        #
        inference_path = os.path.split(work_dir)[-1]
        output_dict.update({'inference_path':inference_path})
        return output_dict

    def _run_session(self, session, data):
        if hasattr(session, 'set_info') and callable(session.set_info):
            session.set_info(self.info_dict)
        #
        output = session.infer_frame(data)
        if hasattr(session, 'get_info') and callable(session.get_info):
            self.info_dict = utils.dict_merge(self.info_dict, session.get_info(), inplace=True)
        #
        return output

    def _sequential_pipeline(self, pipeline, data):
        if pipeline is not None:
            pipeline = utils.as_list(pipeline)
            for pipeline_stage in pipeline:
                if hasattr(pipeline_stage, 'set_info') and callable(pipeline_stage.set_info):
                    pipeline_stage.set_info(self.info_dict)
                #

                data = pipeline_stage(data)

                if hasattr(pipeline_stage, 'get_info') and callable(pipeline_stage.get_info):
                    self.info_dict = utils.dict_merge(self.info_dict, pipeline_stage.get_info(), inplace=True)
                #
            #
        #
        return data

    def _parallel_pipeline(self, pipeline, data):
        if pipeline is not None:
            d_list = []
            for pipeline_stage in pipeline:
                if hasattr(pipeline_stage, 'set_info') and callable(pipeline_stage.set_info):
                    pipeline_stage.set_info(self.info_dict)
                #

                data = pipeline_stage(data)

                if hasattr(pipeline_stage, 'get_info') and callable(pipeline_stage.get_info):
                    self.info_dict = utils.dict_merge(self.info_dict, pipeline_stage.get_info(), inplace=True)
                #
                d_list.append(data)
            #
            data = d_list
        #
        return data
