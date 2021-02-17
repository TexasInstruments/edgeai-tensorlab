import os
import datetime
import shutil
from memory_tempfile import MemoryTempfile
from .. import utils


class BaseRTSession():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tempfiles = []
        self.import_done = False

        self.kwargs['work_dir'] = self.kwargs.get('work_dir', None)
        self.kwargs['model_id'] = self.kwargs.get('model_id', None)
        self.kwargs['dir_tree_depth'] = self.kwargs.get('dir_tree_depth', 2)
        # options related to the underlying runtime
        self.kwargs['platform'] = self.kwargs.get('platform', 'J7')
        self.kwargs['version'] = self.kwargs.get('version', (7,0))
        self.kwargs['tidl_tensor_bits'] = self.kwargs.get('tidl_tensor_bits', 32)
        self.kwargs['num_tidl_subgraphs'] = self.kwargs.get('num_tidl_subgraphs', 16)
        self.kwargs['model_path'] = os.path.abspath(self.kwargs.get('model_path',None))
        self.kwargs['input_shape'] = self.kwargs.get('input_shape', None)
        self.kwargs['num_inputs'] = self.kwargs.get('num_inputs', 1)

        self.kwargs['supported_devices'] = self.kwargs.get('supported_devices', None) #TODO: change to => ('j7', 'pc')
        if self.kwargs['supported_devices'] is not None:
            assert isinstance(self.kwargs['supported_devices'], (list,tuple)), \
                f'supported_device must be a list or tuple'
            assert self.kwargs['target_device'] in self.kwargs['supported_devices'], \
                f"unsupported target device, must be one of {self.kwargs['supported_devices']}"
        #
        self.cwd = os.getcwd()

    def set_param(self, param_name, param):
        self.kwargs[param_name] = param

    def start(self):
        self.kwargs['work_dir'] = self._get_or_make_work_dir()
        artifacts_folder_default = os.path.join(self.kwargs['work_dir'], 'artifacts')
        self.kwargs['artifacts_folder'] = self.kwargs.get('artifacts_folder', artifacts_folder_default)
        os.makedirs(self.kwargs['work_dir'], exist_ok=True)

    def import_model(self, calib_data, info_dict=None):
        os.makedirs(self.kwargs['work_dir'], exist_ok=True)
        os.makedirs(self.kwargs['artifacts_folder'], exist_ok=True)
        model_root_default = os.path.join(self.kwargs['work_dir'], 'model')
        model_path = utils.download_file(self.kwargs['model_path'], root=model_root_default)
        model_path = os.path.abspath(model_path)
        self.kwargs['model_path'] = model_path

    def start_infer(self):
        pass

    def __call__(self, input, info_dict):
        return self.infer_frame(input, info_dict)

    def infer_frame(self, input, info_dict=None):
        assert self.import_done == True, 'the given model must be an imported one.'
        if info_dict is not None:
            info_dict['work_dir'] = self.get_work_dir()
        #

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

    def get_detections(self, **kwargs):
        return None

    def get_work_dir(self):
        return self.kwargs['work_dir']

    def _get_or_make_work_dir(self):
        dir_tree_depth = self.kwargs['dir_tree_depth']
        # MemoryTempfile() creates a file in RAM, which should be really fast.
        if self.kwargs['work_dir'] is None:
            temp_dir_mem = MemoryTempfile()
            # if MemoryTempfile fails it returns a file using tempfile - so we need to check
            temp_dir = temp_dir_mem.tempdir if hasattr(temp_dir_mem, 'tempdir') else temp_dir_mem.name
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
            root_dir = os.path.join(temp_dir, date)
            self.tempfiles.append(root_dir)
        #
        self.kwargs['work_dir'] = os.path.abspath(self.kwargs['work_dir'])
        model_name = self.kwargs['model_path']
        model_name, model_ext = os.path.splitext(model_name)
        model_ext = model_ext[1:]
        model_name_splits = model_name.split(os.sep)
        if len(model_name_splits) > dir_tree_depth:
            model_name_splits = model_name_splits[-dir_tree_depth:]
        #
        model_id = self.kwargs['model_id']
        model_name = '_'.join(model_name_splits + [model_ext])
        session_name = self.kwargs['session_name']
        work_dir = os.path.join(self.kwargs['work_dir'], f'{model_id}_{model_name}_{session_name}')
        return work_dir

    def _dict_equal(self, shape1, shape2):
        for k1, v1 in shape1.items():
            if k1 not in shape2:
                return False
            #
            v2 = shape2[k1]
            if isinstance(v1, (list,tuple)) or isinstance(v2, (list,tuple)):
                if any(v1 != v2):
                    return False
                #
            elif v1 != v2:
                return False
            #
        #
        return True


if __name__ == '__main__':
    import_model = BaseRTSession()
