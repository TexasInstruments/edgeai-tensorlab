import os
import datetime
import shutil
from memory_tempfile import MemoryTempfile
from .. import utils

class BaseRTSession():
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs['platform'] = self.kwargs.get('platform', 'J7')
        self.kwargs['version'] = self.kwargs.get('version', (7,0))
        self.kwargs['tidl_tensor_bits'] = self.kwargs.get('tidl_tensor_bits', 32)
        self.kwargs['num_tidl_subgraphs'] = self.kwargs.get('num_tidl_subgraphs', 16)
        self.kwargs['work_dir'] = self._get_or_make_work_dir()
        self.artifacts_folder = os.path.join(self.kwargs['work_dir'], 'artifacts')
        self.kwargs['model_path'] = os.path.normpath(self.kwargs.get('model_path',None))
        self.kwargs['input_shape'] = self.kwargs.get('input_shape', None)
        self.import_done = False

    def import_model(self, calib_data):
        os.makedirs(self.kwargs['work_dir'], exist_ok=True)
        self.kwargs['model_path'] = utils.download_model(self.kwargs['model_path'],
            root=os.path.join(self.kwargs['work_dir'],'model'))

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
        self.kwargs['work_dir'] = os.path.normpath(self.kwargs['work_dir'])
        model_name = self.kwargs['model_path']
        model_name = os.path.splitext(model_name)[0]
        model_name = '_'.join(model_name.split(os.sep)[-dir_tree_depth:])
        work_dir = os.path.join(self.kwargs['work_dir'], model_name)
        return work_dir

if __name__ == '__main__':
    import_model = BaseRTSession()
