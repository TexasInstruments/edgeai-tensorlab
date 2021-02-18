import os
import contextlib
from .. import utils


class AccuracyPipeline():
    def __init__(self, pipeline_config):
        self.info_dict = dict()
        self.pipeline_config = pipeline_config
        self.logger = None

    def __del__(self):
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        #

    def run(self):
        # run the actual model
        result = {}
        run_import = self.pipeline_config['run_import']
        run_inference = self.pipeline_config['run_inference']
        session = self.pipeline_config['session']

        # start must be called to create the work directory
        session.start()

        # logger can be created after start
        work_dir = self.pipeline_config['session'].get_param('work_dir')
        # verbose = self.pipeline_config['verbose']
        file_name = os.path.join(work_dir, 'run.log')
        self.logger = utils.TeeLogger(file_name)
        self.logger.write(f'\npipeline_config: {self.pipeline_config}')

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
        description = os.path.split(session.get_param('work_dir'))[-1]
        self.logger.write('import & calibration: ' + description)

        calib_data = []
        num_frames = len(calibration_dataset)
        for data_index in range(num_frames):
            info_dict = {}
            data = calibration_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)
            calib_data.append(data)
        #
        session.import_model(calib_data)

    def _infer_frames(self):
        session = self.pipeline_config['session']
        input_dataset = self.pipeline_config['input_dataset']
        preprocess = self.pipeline_config['preprocess']
        postprocess = self.pipeline_config['postprocess']
        description = os.path.split(session.get_param('work_dir'))[-1]

        is_ok = session.start_infer()
        assert is_ok, f'start_infer() did not succeed for {description}'

        output_list = []
        num_frames = len(input_dataset)
        for data_index in utils.progress_step(range(num_frames), desc='infer: '+description, file=self.logger):
            info_dict = {}
            data = input_dataset[data_index]
            data, info_dict = preprocess(data, info_dict)
            output, info_dict = session.infer_frame(data, info_dict)
            output, info_dict = postprocess(output, info_dict)
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
        work_dir = session.get_param('work_dir')
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
