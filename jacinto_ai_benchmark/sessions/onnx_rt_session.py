import os
import time
import numpy as np
import onnxruntime
from .. import utils
from .base_rt_session import BaseRTSession

class ONNXRTSession(BaseRTSession):
    def __init__(self, session_name='onnxrt', **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.kwargs['output_shape'] = self.kwargs.get('output_shape', None)
        self.interpreter = None

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)
        self.interpreter = None
        return info_dict

    def start_infer(self):
        if not os.path.exists(self.kwargs['artifacts_folder']):
            return False
        #
        super().start_infer()
        # now create the interpreter for inference
        self.interpreter = onnxruntime.InferenceSession(self.kwargs['model_path'], None)
        if self.kwargs['input_shape'] is None:
            input_shape = self._get_input_shape_onnxrt()
            self.kwargs['input_shape'] = input_shape
        #
        if self.kwargs['output_shape'] is None:
            output_shape = self._get_output_shape_onnxrt()
            self.kwargs['output_shape'] = output_shape
        #
        os.chdir(self.cwd)
        self.is_imported = True
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)

        input_shape = self.kwargs['input_shape']
        input_keys = list(input_shape.keys())
        in_data = utils.as_tuple(input)
        input_dict = {d_name:d for d_name, d in zip(input_keys,in_data)}

        # model needs additional inputs given in extra_inputs
        if 'extra_inputs' in self.kwargs:
            extra_inputs = self.kwargs['extra_inputs']
            input_dict.update(extra_inputs)
        #

        output_shape = self.kwargs['output_shape']
        output_keys = list(output_shape.keys())

        # run the actual inference
        start_time = time.time()
        outputs = self.interpreter.run(output_keys, input_dict)
        info_dict['session_invoke_time'] = (time.time() - start_time)
        return outputs, info_dict

    def _set_default_options(self):
        pass

    def _get_input_shape_onnxrt(self):
        input_details = self.interpreter.get_inputs()
        input_shape = {}
        for inp in input_details:
            input_shape.update({inp.name:inp.shape})
        #
        return input_shape

    def _get_output_shape_onnxrt(self):
        output_details = self.interpreter.get_outputs()
        output_shape = {}
        for oup in output_details:
            output_shape.update({oup.name:oup.shape})
        #
        return output_shape
