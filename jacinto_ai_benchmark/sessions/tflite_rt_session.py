import os
import numpy as np
import tflite_runtime.interpreter as tflitert_interpreter
from .. import utils
from .base_rt_session import BaseRTSession

class TFLiteRTSession(BaseRTSession):
    def __init__(self, session_name='tflite-rt', **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self._set_default_options()
        self.interpreter = None
        self.interpreter_folder = os.path.join(os.environ['TIDL_BASE_PATH'], 'ti_dl/test/tflrt')

    def import_model(self, calib_data):
        super().import_model(calib_data)
        os.chdir(self.interpreter_folder)
        self.interpreter = self._create_interpreter(is_import=True)

        # check if the shape of data being proved matches with what model expects
        data_input_shape = self.kwargs['input_shape']
        model_input_details = self.interpreter.get_input_details()
        model_output_details = self.interpreter.get_output_details()
        for model_input_idx, model_input in enumerate(model_input_details):
            model_input_name = model_input['name']
            model_input_shape = model_input['shape']
            assert model_input_name in data_input_shape, f'could not find {model_input_name} in the keys of input_shape dict provided {data_input_shape}'
            tensor_input_shape = data_input_shape[model_input_name]
            assert any(model_input_shape == tensor_input_shape), f'invalid input tensor height: expected {model_input_shape} obtained {data_input_shape}'

        for c_data in calib_data:
            c_data = utils.as_tuple(c_data)
            for c_data_entry_idx, c_data_entry in enumerate(c_data):
                self.interpreter.set_tensor(model_input_details[c_data_entry_idx]['index'], c_data_entry)
            #
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(output_detail['index']) for output_detail in model_output_details]
        #

    def start_infer(self):
        super().start_infer()
        # now create the interpreter for inference
        os.chdir(self.interpreter_folder)
        self.interpreter = self._create_interpreter(is_import=False)
        os.chdir(self.cwd)
        self.import_done = True

    def infer_frame(self, input):
        super().infer_frame(input)
        model_input_details = self.interpreter.get_input_details()
        model_output_details = self.interpreter.get_output_details()
        c_data = utils.as_tuple(input)
        for c_data_entry_idx, c_data_entry in enumerate(c_data):
            self.interpreter.set_tensor(model_input_details[c_data_entry_idx]['index'], c_data_entry)
        #
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(output_detail['index']) for output_detail in model_output_details]
        return outputs

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
