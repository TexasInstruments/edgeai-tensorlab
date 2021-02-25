import os
import time
import warnings
import numpy as np
import tflite_runtime.interpreter as tflitert_interpreter
from .. import utils
from .base_rt_session import BaseRTSession


class TFLiteRTSession(BaseRTSession):
    def __init__(self, session_name='tflitert', **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.interpreter = None
        self.interpreter_folder = os.path.join(os.environ['TIDL_BASE_PATH'], 'ti_dl/test/tflrt')

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)
        os.chdir(self.interpreter_folder)
        self.interpreter = self._create_interpreter(is_import=True)

        # check if the shape of data being proved matches with what model expects
        input_shape = self._get_input_shape_tflite()
        if (self.kwargs['input_shape'] is not None) and (not utils.dict_equal(input_shape, self.kwargs['input_shape'])):
            warnings.warn('model input shape must match the provided shape')
        #

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        for c_data in calib_data:
            c_data = utils.as_tuple(c_data)
            for c_data_entry_idx, c_data_entry in enumerate(c_data):
                self._set_tensor(input_details[c_data_entry_idx], c_data_entry)
            #
            self.interpreter.invoke()
            outputs = [self._get_tensor(output_detail) for output_detail in output_details]
        #
        return info_dict

    def start_infer(self):
        if not os.path.exists(self.kwargs['artifacts_folder']):
            return False
        #
        super().start_infer()
        # now create the interpreter for inference
        os.chdir(self.interpreter_folder)
        self.interpreter = self._create_interpreter(is_import=False)
        os.chdir(self.cwd)
        self.is_imported = True
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        c_data = utils.as_tuple(input)
        for c_data_entry_idx, c_data_entry in enumerate(c_data):
            self._set_tensor(input_details[c_data_entry_idx], c_data_entry)
        #
        # measure the time across only interpreter.run
        # time for setting the tensor and other overheads would be optimized out in c-api
        start_time = time.time()
        self.interpreter.invoke()
        info_dict['session_invoke_time'] = (time.time() - start_time)
        outputs = [self._get_tensor(output_detail) for output_detail in output_details]
        return outputs, info_dict

    def _create_interpreter(self, is_import):
        self.kwargs["delegate_options"]["import"] = "yes" if is_import else "no"
        tidl_delegate = [tflitert_interpreter.load_delegate('libtidl_tfl_delegate.so.1.0', self.kwargs["delegate_options"])]
        interpreter = tflitert_interpreter.Interpreter(model_path=self.kwargs['model_path'], experimental_delegates=tidl_delegate)
        interpreter.allocate_tensors()
        return interpreter

    def _set_default_options(self):
        delegate_options = self.kwargs.get("delegate_options", {})
        tidl_tools_path = os.path.join(os.environ['TIDL_BASE_PATH'], 'tidl_tools')
        required_options = {
            "tidl_tools_path": self.kwargs.get("tidl_tools_path", tidl_tools_path),
            "artifacts_folder": self.kwargs['artifacts_folder'],
            "import": self.kwargs.get("import", 'no')
        }

        optional_options = {
            "tidl_platform": "J7",
            "tidl_version": "7.2",
            "tidl_tensor_bits": self.kwargs.get("tidl_tensor_bits", 32),
            "debug_level": self.kwargs.get("debug_level", 0),
            "num_tidl_subgraphs": self.kwargs.get("num_tidl_subgraphs", 16),
            "tidl_denylist": self.kwargs.get("tidl_denylist", ""),
            "tidl_calibration_method": self.kwargs.get("tidl_calibration_method", "advanced"),
        }

        tidl_calibration_options = self.kwargs.get("tidl_calibration_options", {})
        delegate_options.update(tidl_calibration_options)
        delegate_options.update(required_options)
        delegate_options.update(optional_options)
        self.kwargs["delegate_options"] = delegate_options

    def _get_input_shape_tflite(self):
        input_shape = {}
        model_input_details = self.interpreter.get_input_details()
        for model_input in model_input_details:
            name = model_input['name']
            shape = model_input['shape']
            input_shape.update({name:shape})
        #
        return input_shape

    def _set_tensor(self, model_input, tensor):
        if model_input['dtype'] == np.int8:
            scale, zero_point = model_input['quantization']
            tensor = np.clip(np.round(tensor/scale + zero_point), -128, 127)
            tensor = np.array(tensor, dtype=np.int8)
        elif model_input['dtype'] == np.uint8:
            scale, zero_point = model_input['quantization']
            tensor = np.clip(np.round(tensor/scale + zero_point), 0, 255)
            tensor = np.array(tensor, dtype=np.uint8)
        #
        self.interpreter.set_tensor(model_input['index'], tensor)


    def _get_tensor(self, model_output):
        tensor = self.interpreter.get_tensor(model_output['index'])
        if model_output['dtype'] == np.int8 or model_output['dtype']  == np.uint8:
            scale, zero_point = model_output['quantization']
            tensor = np.array(tensor, dtype=np.float32)
            tensor = (tensor - zero_point) / scale
        #
        return tensor
