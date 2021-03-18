# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import datetime
import shutil
from memory_tempfile import MemoryTempfile
import re
import csv
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
        self.kwargs['artifacts_folder'] = os.path.join(self.kwargs['run_dir'], 'artifacts')
        self.kwargs['model_folder'] = os.path.join(self.kwargs['run_dir'], 'model')
        self._set_default_options()
        super().initialize()

    def start(self):
        assert self.is_initialized, 'initialize() must be called before start_import()'
        os.makedirs(self.kwargs['run_dir'], exist_ok=True)
        os.makedirs(self.kwargs['artifacts_folder'], exist_ok=True)
        self._get_model()
        self.is_started = True

    def import_model(self, calib_data, info_dict=None):
        assert self.is_initialized, 'initialize() must be called before import_model()'
        assert self.is_started, 'start() must be called before import_model()'

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

    def __del__(self):
        for t in self.tempfiles:
            if os.path.exists(t):
                shutil.rmtree(t)

    def infer_stats(self):
        if hasattr(self.interpreter, 'get_TI_benchmark_data'):
            stats_dict = self._tidl_infer_stats()
        else:
            stats_dict = dict()
            stats_dict['num_subgraphs'] = 0
            stats_dict['core_time'] = 0.0
            stats_dict['subgraph_time'] = 0.0
            stats_dict['read_total'] = 0.0
            stats_dict['write_total'] = 0.0
        #
        return stats_dict

    def _tidl_infer_stats(self):
        assert self.is_imported == True, 'the given model must be an imported one.'
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
        total_time = total_time / constants.DSP_FREQ
        copy_time = copy_time / constants.DSP_FREQ
        subgraph_time = subgraph_time / constants.DSP_FREQ
        write_total = write_total
        read_total = read_total
        # core time excluding the copy overhead
        core_time = total_time - copy_time
        stats = {
            'num_subgraphs': len(subgraphIds),
            'total_time': total_time, 'core_time': core_time, 'subgraph_time': subgraph_time,
            'write_total': write_total, 'read_total': read_total
        }
        try:
            perfsim_stats = self._infer_perfsim_stats()
            stats.update(perfsim_stats)
        except:
            pass
        #
        return stats

    def _infer_perfsim_stats(self):
        assert self.is_imported == True, 'the given model must be an imported one.'
        artifacts_folder = self.kwargs['artifacts_folder']
        subgraph_root = os.path.join(artifacts_folder, 'tempDir') \
            if os.path.isdir(os.path.join(artifacts_folder, 'tempDir')) else artifacts_folder
        perfsim_folders = [os.path.join(subgraph_root, d) for d in os.listdir(subgraph_root)]
        perfsim_folders = [d for d in perfsim_folders if os.path.isdir(d)]
        perfsim_dict = {}
        for perfsim_folder in perfsim_folders:
            subgraph_stats = self._subgraph_perfsim_stats(perfsim_folder)
            for k, v in subgraph_stats.items():
                if k in perfsim_dict:
                    perfsim_dict[k] += v
                else:
                    perfsim_dict[k] = v
                #
            #
        #
        return perfsim_dict
    #

    def _subgraph_perfsim_stats(self, perfsim_folder):
        perfsim_files = os.listdir(perfsim_folder)
        if len(perfsim_files) == 0:
            return None
        #
        subgraph_perfsim_dict = {}
        # get the gmac number from netLog file
        netlog_file = perfsim_folder + '.bin_netLog.txt'
        with open(netlog_file) as netlog_fp:
            netlog_reader = csv.reader(netlog_fp)
            netlog_data = [data for data in netlog_reader]
            perfsim_macs = [row for row in netlog_data if 'total giga macs' in row[0].lower()][0][0]
            perfsim_macs = float(perfsim_macs.split(':')[1])
            # change units - convert gmacs to macs
            perfsim_macs = perfsim_macs * constants.GIGA_CONST
            subgraph_perfsim_dict.update({'perfsim_macs': perfsim_macs})
        #
        # get the perfsim cycles
        graph_name = os.path.basename(perfsim_folder)
        perfsim_csv = [p for p in perfsim_files if graph_name in p and os.path.splitext(p)[1] == '.csv' and not p.startswith('.')][0]
        perfsim_csv = os.path.join(perfsim_folder, perfsim_csv)
        with open(perfsim_csv) as perfsim_fp:
            perfsim_reader = csv.reader(perfsim_fp)
            perfsim_data = [data for data in perfsim_reader]

            # perfsim time - read from file
            perfsim_time = [row for row in perfsim_data if 'total network time (us)' in row[0].lower()][0][0]
            perfsim_time = float(perfsim_time.split('=')[1])
            # change units - convert from ultrasec to seconds
            perfsim_time = perfsim_time / constants.ULTRA_CONST
            subgraph_perfsim_dict.update({'perfsim_time': perfsim_time})

            # perfsim cycles - read from file
            # perfsim_cycles = [row for row in perfsim_data if 'total network cycles (mega)' in row[0].lower()][0][0]
            # perfsim_cycles = float(perfsim_cycles.split('=')[1])
            # change units - convert from mega cycles to cycles
            # perfsim_cycles = perfsim_cycles * constants.MEGA_CONST
            # subgraph_perfsim_dict.update({'perfsim_cycles': perfsim_cycles})

            # perfsim ddr transfer - read from file
            perfsim_ddr_transfer = [row for row in perfsim_data if 'ddr bw (mega bytes) : total' in row[0].lower()][0][0]
            perfsim_ddr_transfer = float(perfsim_ddr_transfer.split('=')[1])
            # change units - convert from megabytes to bytes
            perfsim_ddr_transfer = perfsim_ddr_transfer * constants.MEGA_CONST
            subgraph_perfsim_dict.update({'perfsim_ddr_transfer': perfsim_ddr_transfer})
        #
        return subgraph_perfsim_dict

    def _make_run_dir(self):
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
        run_name = '_'.join([model_id] + [session_name] + model_name_splits + [model_ext])
        run_dir = os.path.join(work_dir, f'{run_name}')
        return run_dir

    def _get_model(self):
        # download the file if it is an http or https link
        model_folder = self.kwargs['model_folder']
        model_path = utils.download_file(self.kwargs['model_path'], root=model_folder)
        # make a local copy
        model_path_local = utils.get_local_path(model_path, model_folder)
        if not utils.file_exists(model_path_local):
            utils.copy_files(model_path, model_path_local)
        #
        self.kwargs['model_path'] = model_path_local

    def _set_default_options(self):
        assert False, 'this function must be overridden in the derived class'


if __name__ == '__main__':
    import_model = BaseRTSession()
