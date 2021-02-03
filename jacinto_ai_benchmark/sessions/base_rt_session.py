import os
import datetime
import shutil
from memory_tempfile import MemoryTempfile
from .. import utils

class BaseRTSession():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.import_done = False
        self.kwargs['work_dir'] = self._get_or_make_work_dir()
        # options related to the underlying runtime
        self.kwargs['platform'] = self.kwargs.get('platform', 'J7')
        self.kwargs['version'] = self.kwargs.get('version', (7,0))
        self.kwargs['tidl_tensor_bits'] = self.kwargs.get('tidl_tensor_bits', 32)
        self.kwargs['num_tidl_subgraphs'] = self.kwargs.get('num_tidl_subgraphs', 16)
        artifacts_folder_default = os.path.join(self.kwargs['work_dir'], 'artifacts')
        self.kwargs['artifacts_folder'] = self.kwargs.get('artifacts_folder', artifacts_folder_default)
        self.kwargs['model_path'] = os.path.abspath(self.kwargs.get('model_path',None))
        self.kwargs['input_shape'] = self.kwargs.get('input_shape', None)

    def import_model(self, calib_data):
        os.makedirs(self.kwargs['work_dir'], exist_ok=True)
        os.makedirs(self.kwargs['artifacts_folder'], exist_ok=True)
        model_root_default = os.path.join(self.kwargs['work_dir'], 'model')
        model_path = utils.download_model(self.kwargs['model_path'], root=model_root_default)
        model_path = os.path.abspath(model_path)
        self.kwargs['model_path'] = model_path
        self.cwd = os.getcwd()

    def __call__(self, input):
        return self.infer_frame(input)

    def infer_frame(self, input):
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
        self.kwargs['work_dir'] = os.path.abspath(self.kwargs['work_dir'])
        model_name = self.kwargs['model_path']
        model_name = os.path.splitext(model_name)[0]
        model_name = '_'.join(model_name.split(os.sep)[-dir_tree_depth:])
        session_name = self.kwargs['session_name']
        tidl_tensor_bits = self.kwargs['tidl_tensor_bits']
        work_dir = os.path.join(self.kwargs['work_dir'], f'{session_name}_{model_name}')
        return work_dir


if __name__ == '__main__':
    import_model = BaseRTSession()
