import os
import tqdm
from .. import utils

class AccuracyPipeline():
    def __init__(self, pipeline_config):
        self.info_dict = dict()
        self.pipeline_config = pipeline_config
        work_dir = self.pipeline_config['session'].get_work_dir()
        os.makedirs(work_dir, exist_ok=True)
        # after the logger is created, print is supposed to write to file and term
        # but sometimes it doesn't work - so use logger.write() to enforce it wherever needed.
        self.logger = utils.TeeLogger(os.path.join(work_dir, 'run.log'))
        self.logger.write(f'\npipeline_config: {self.pipeline_config}')

    def run(self):
        result = {}

        run_import = self.pipeline_config['run_import']
        run_inference = self.pipeline_config['run_inference']

        if run_import:
            self._import_model()
        #
        if run_inference:
            output_list = self._infer_frames()
            result = self._evaluate(output_list)
        #

        self.logger.write(f'\nBenchmarkResults: {result}')
        return result

    def _import_model(self):
        session = self.pipeline_config['session']
        calibration_dataset = self.pipeline_config['calibration_dataset']
        preprocess = self.pipeline_config['preprocess']
        description = os.path.split(session.get_work_dir())[-1]
        self.logger.write('import & calibration: ' + description)

        calib_data = []
        num_frames = len(calibration_dataset)
        for data_index in range(num_frames):
            data = calibration_dataset[data_index]
            data = self._sequential_pipeline(preprocess, data)
            calib_data.append(data)
        #

        session.import_model(calib_data)

    def _infer_frames(self):
        session = self.pipeline_config['session']
        input_dataset = self.pipeline_config['input_dataset']
        preprocess = self.pipeline_config['preprocess']
        postprocess = self.pipeline_config['postprocess']
        description = os.path.split(session.get_work_dir())[-1]

        is_ok = session.start_infer()
        assert is_ok, f'start_infer() did not succeed for {description}'

        output_list = []
        num_frames = len(input_dataset)
        for data_index in utils.progress_step(range(num_frames), desc='infer: '+description, file=self.logger):
            data = input_dataset[data_index]
            data = self._sequential_pipeline(preprocess, data)
            output = self._run_session_with_data(session, data)
            output = self._sequential_pipeline(postprocess, output)
            output_list.append(output)
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

    def _run_session_with_data(self, session, data):
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
