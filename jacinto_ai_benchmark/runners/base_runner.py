import os
import datetime
import shutil
from memory_tempfile import MemoryTempfile
from .. import utils

class BaseRunner():
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs['platform'] = self.kwargs.get('platform', 'J7')
        self.kwargs['version'] = self.kwargs.get('version', (7,0))
        self.kwargs['tidl_tensor_bits'] = self.kwargs.get('tidl_tensor_bits', 32)
        self.kwargs['data_layout'] = self.kwargs.get('data_layout', 'NCHW')
        self.kwargs['num_tidl_subgraphs'] = self.kwargs.get('num_tidl_subgraphs', 16)
        self.kwargs['tidl_calibration_options'] = self._get_calibration_options(
            **self.kwargs['tidl_calibration_options'])
        self.kwargs['work_dir'] = self._get_or_make_work_dir()
        self.artifacts_folder = os.path.join(self.kwargs['work_dir'], 'artifacts')
        self.kwargs['model_path'] = os.path.normpath(self.kwargs.get('model_path',None))

    def import_model(self, calib_data):
        os.makedirs(self.kwargs['work_dir'], exist_ok=True)
        self.kwargs['model_path'] = utils.download_model(self.kwargs['model_path'],
            root=os.path.join(self.kwargs['work_dir'],'model'))

    def __call__(self, **kwargs):
        return self.infer_frame(**kwargs)

    def infer_frame(self, **kwargs):
        pass

    def __del__(self):
        for t in self.tempfiles:
            if os.path.exists(t):
                shutil.rmtree(t)

    def perfsim_data(self):
        assert self.import_done == True, 'the given model must be an imported one.'
        return None

    def infer_layers(self, **kwargs):
        return None

    def layer_names(self):
        return None

    def get_detections(self, output_array, inResizeType=0):
        return None

    def get_work_dir(self):
        return self.kwargs['work_dir']

    def _get_or_make_work_dir(self, dir_tree_depth=3):
        self.tempfiles = []
        # MemoryTempfile() creates a file in RAM, which should be really fast.
        if self.kwargs['work_dir'] is None:
            temp_dir_mem = MemoryTempfile()
            # if MemoryTempfile fails it returns a file using tempfile - so we need to check
            temp_dir = temp_dir_mem.tempdir if hasattr(temp_dir_mem, 'temp_dir') else temp_dir_mem.name
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
            root_dir = os.path.join(temp_dir, date)
            self.tempfiles.append(root_dir)
        #
        model_name = self.kwargs['model_path']
        model_name = os.path.splitext(model_name)[0]
        model_name = '_'.join(model_name.split(os.sep)[-dir_tree_depth:])
        work_dir = os.path.join(self.kwargs['work_dir'], model_name)
        return work_dir

    def _as_list(self, arg):
        return arg if isinstance(arg, list) else [arg]

    def _get_calibration_options(self, **kwargs_calib):
        kwargs_calib['activation_range'] = kwargs_calib.get('activation_range', 'on')
        kwargs_calib['weight_range'] = kwargs_calib.get('weight_range', 'on')
        kwargs_calib['bias_calibration'] = kwargs_calib.get('bias_calibration', 'on')
        kwargs_calib['per_channel_weight'] = kwargs_calib.get('per_channel_weight', 'off')
        kwargs_calib['iterations'] = kwargs_calib.get('iterations', 50)
        kwargs_calib['num_frames'] = kwargs_calib.get('num_frames', 100)
        return kwargs_calib

if __name__ == '__main__':
    import_model = BaseRunner()
