import os
import datetime
import shutil
from memory_tempfile import MemoryTempfile
import re
from .. import utils
from .. import constants


class BaseRTSession(utils.ParamsBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tempfiles = []
        self.is_initialized = False
        self.is_started = False
        self.is_imported = False
        # work_dir at top level
        self.kwargs['work_dir'] = self.kwargs.get('work_dir', None)
        # run_dir for individual model
        self.kwargs['run_dir'] = self.kwargs.get('run_dir', None)
        self.kwargs['dir_tree_depth'] = self.kwargs.get('dir_tree_depth', 2)
        # options related to the underlying runtime
        self.kwargs['platform'] = self.kwargs.get('platform', 'J7')
        self.kwargs['version'] = self.kwargs.get('version', (7,0))
        self.kwargs['tidl_tensor_bits'] = self.kwargs.get('tidl_tensor_bits', 32)
        self.kwargs['num_tidl_subgraphs'] = self.kwargs.get('num_tidl_subgraphs', 16)
        self.kwargs['model_id'] = self.kwargs.get('model_id', None)
        # convert model_path to abspath
        model_path = self.kwargs.get('model_path', None)
        model_path = [os.path.abspath(m) for m in model_path] if isinstance(model_path, (list,tuple)) else model_path
        self.kwargs['model_path'] = os.path.abspath(model_path) if isinstance(model_path, str) else model_path
        self.kwargs['model_type'] = self.kwargs.get('model_type',None)
        self.kwargs['input_shape'] = self.kwargs.get('input_shape', None)
        self.kwargs['num_inputs'] = self.kwargs.get('num_inputs', 1)
        # check the target_device
        self.kwargs['supported_devices'] = self.kwargs.get('supported_devices', None) #TODO: change to => ('j7', 'pc')
        if self.kwargs['supported_devices'] is not None:
            assert isinstance(self.kwargs['supported_devices'], (list,tuple)), \
                f'supported_device must be a list or tuple'
            assert self.kwargs['target_device'] in self.kwargs['supported_devices'], \
                f"unsupported target device, must be one of {self.kwargs['supported_devices']}"
        #
        # store the current directory so that we can go back there any time
        self.cwd = os.getcwd()

    def initialize(self):
        # make run_dir path
        self.kwargs['run_dir'] = self._make_run_dir()
        artifacts_folder_default = os.path.join(self.kwargs['run_dir'], 'artifacts')
        self.kwargs['artifacts_folder'] = self.kwargs.get('artifacts_folder', artifacts_folder_default)
        self._set_default_options()
        super().initialize()

    def start(self):
        assert self.is_initialized, 'initialize() must be called before start_import()'
        os.makedirs(self.kwargs['run_dir'], exist_ok=True)
        os.makedirs(self.kwargs['artifacts_folder'], exist_ok=True)
        self.is_started = True

    def import_model(self, calib_data, info_dict=None):
        assert self.is_initialized, 'initialize() must be called before import_model()'
        assert self.is_started, 'start() must be called before import_model()'
        model_root_default = os.path.join(self.kwargs['run_dir'], 'model')
        model_root_default = os.path.abspath(model_root_default)
        model_path = utils.download_file(self.kwargs['model_path'], root=model_root_default)
        self.kwargs['model_path'] = model_path

    def start_infer(self):
        pass

    def __call__(self, input, info_dict):
        return self.infer_frame(input, info_dict)

    def infer_frame(self, input, info_dict=None):
        assert self.is_initialized, 'initialize() must be called before before infer_frame()'
        assert self.is_started, 'start() must be called before before infer_frame()'
        assert self.is_imported, 'import_model() and start_infer() must be called before infer_frame()'
        if info_dict is not None:
            info_dict['run_dir'] = self.get_param('run_dir')
        #

    def infer_stats(self):
        benchmark_dict = self.interpreter.get_TI_benchmark_data()
        subgraph_time = copy_time = 0
        cp_in_time = cp_out_time = 0
        subgraphIds = []
        for stat in benchmark_dict.keys():
            if 'proc_start' in stat:
                subgraphIds.append(int(re.sub("[^0-9]", "", stat)))
            #
        #
        for i in range(len(subgraphIds)):
            subgraph_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
            cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
            cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
        #
        copy_time = cp_in_time + cp_out_time
        copy_time = copy_time if len(subgraphIds) == 1 else 0
        total_time = benchmark_dict['ts:run_end'] - benchmark_dict['ts:run_start']
        write_total = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
        read_total = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']
        # change units
        total_time = total_time/constants.PROCESSOR_FREQ
        copy_time = copy_time/constants.PROCESSOR_FREQ
        subgraph_time = subgraph_time/constants.PROCESSOR_FREQ
        # core time excluding the copy overhead
        core_time = total_time - copy_time
        stats = {
            'num_subgraphs': len(subgraphIds),
            'total_time': total_time, 'core_time': core_time, 'subgraph_time': subgraph_time,
            'write_total': write_total, 'read_total': read_total
        }
        return stats

    def __del__(self):
        for t in self.tempfiles:
            if os.path.exists(t):
                shutil.rmtree(t)

    def perfsim_data(self):
        assert self.is_imported == True, 'the given model must be an imported one.'
        return None

    def infer_layers(self, **kwargs):
        return None

    def layer_names(self):
        return None

    def get_detections(self, **kwargs):
        return None

    def _make_run_dir(self):
        if self.kwargs['run_dir'] is not None:
            return self.kwargs['run_dir']
        #
        # MemoryTempfile() creates a file in RAM, which should be really fast.
        work_dir = self.kwargs['work_dir']
        if work_dir is None:
            temp_dir_mem = MemoryTempfile()
            # if MemoryTempfile fails it returns a file using tempfile - so we need to check
            temp_dir = temp_dir_mem.tempdir if hasattr(temp_dir_mem, 'tempdir') else temp_dir_mem.name
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
            work_dir = os.path.join(temp_dir, date)
            self.tempfiles.append(work_dir)
        #
        work_dir = os.path.abspath(work_dir)
        model_path = self.kwargs['model_path']
        model_name = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
        model_name, model_ext = os.path.splitext(model_name)
        model_ext = model_ext[1:] if len(model_ext)>0 else model_ext

        model_name_splits = model_name.split(os.sep)
        dir_tree_depth = self.kwargs['dir_tree_depth']
        if len(model_name_splits) > dir_tree_depth:
            model_name_splits = model_name_splits[-dir_tree_depth:]
        #
        model_id = self.kwargs['model_id']
        session_name = self.kwargs['session_name']
        run_name = '_'.join([model_id] + model_name_splits + [model_ext] + [session_name])
        run_dir = os.path.join(work_dir, f'{run_name}')
        return run_dir

    def _set_default_options(self):
        assert False, 'this function must be overridden in the derived class'


if __name__ == '__main__':
    import_model = BaseRTSession()
