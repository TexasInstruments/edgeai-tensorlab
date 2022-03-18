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
import tempfile
import re
import csv
import itertools
import copy
from colorama import Fore
import numpy as np
import tarfile
from .. import utils
from .. import constants
from ..preprocess.transforms import ImageNormMeanScale


class BaseRTSession(utils.ParamsBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tempfiles = []
        self.is_initialized = False
        self.is_started = False
        self.is_imported = False
        self.is_start_infer_done = False
        self.input_normalizer = None

        # set tidl_offload to False to disable offloading to TIDL
        self.kwargs['tidl_offload'] = self.kwargs.get('tidl_offload', True)

        # tidl_tools_path
        assert 'TIDL_TOOLS_PATH' in os.environ, 'TIDL_TOOLS_PATH must be set in environemnt variable'
        tidl_tools_path = os.environ['TIDL_TOOLS_PATH']
        self.kwargs['tidl_tools_path'] = tidl_tools_path

        # work_dir at top level
        self.kwargs['work_dir'] = self.kwargs.get('work_dir', None)
        # run_dir for individual model
        self.kwargs['run_dir'] = self.kwargs.get('run_dir', None)
        self.kwargs['dir_tree_depth'] = self.kwargs.get('dir_tree_depth', 3)

        # parameters related to models
        self.kwargs['num_tidl_subgraphs'] = self.kwargs.get('num_tidl_subgraphs', 16)
        self.kwargs['model_id'] = self.kwargs.get('model_id', None)
        model_path = self.kwargs.get('model_path', None)
        model_path = [os.path.abspath(m) for m in model_path] if isinstance(model_path, (list,tuple)) else model_path
        self.kwargs['model_path'] = os.path.abspath(model_path) if isinstance(model_path, str) else model_path
        self.kwargs['model_type'] = self.kwargs.get('model_type',None)
        self.kwargs['input_shape'] = self.kwargs.get('input_shape', None)
        self.kwargs['output_shape'] = self.kwargs.get('output_shape', None)
        self.kwargs['num_inputs'] = self.kwargs.get('num_inputs', 1)
        self.kwargs['extra_inputs'] = self.kwargs.get('extra_inputs', None)
        # parameters for input optimization
        self.kwargs['input_optimization'] = self.kwargs.get('input_optimization', None)
        self.kwargs['input_mean'] = self.kwargs.get('input_mean', None)
        self.kwargs['input_scale'] = self.kwargs.get('input_scale', None)

        # other parameters
        self.kwargs['tensor_bits'] = self.kwargs.get('tensor_bits', 8)

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
        super().initialize()

    def start(self):
        if not self.is_initialized:
            self.initialize()
        #
        # _set_default_options requires to know the artifacts_folder
        # that's why this is not done in the constructor
        self._set_default_options()

        # if the run_dir doesn't exist, check if tarfile exists or can be downloaded/untarred
        run_dir = self.kwargs['run_dir']
        if not os.path.exists(run_dir):
            work_dir = os.path.dirname(run_dir)
            tarfile_name = run_dir + '.tar.gz'
            if not os.path.exists(tarfile_name):
                tarfile_name = utils.download_file(tarfile_name, work_dir, extract_root=run_dir)
            #
            # extract the tar file
            if (not os.path.exists(run_dir)) and tarfile_name is not None and os.path.exists(tarfile_name):
                os.makedirs(run_dir, exist_ok=True)
                tfp = tarfile.open(tarfile_name)
                tfp.extractall(run_dir)
                tfp.close()
            #
        #

        # create run_dir
        os.makedirs(self.kwargs['run_dir'], exist_ok=True)
        os.makedirs(self.kwargs['model_folder'], exist_ok=True)
        self.get_model()
        self.is_started = True

    def import_model(self, calib_data, info_dict=None):
        if not self.is_initialized:
            self.initialize()
        #
        if not self.is_started:
            self.start()
        #
        os.makedirs(self.kwargs['artifacts_folder'], exist_ok=True)

        self.clear()
        self.is_imported = True

    def start_infer(self):
        artifacts_folder = self.kwargs['artifacts_folder']
        artifacts_folder_missing = not os.path.exists(artifacts_folder)
        if artifacts_folder_missing:
            error_message = utils.log_color('ERROR', f'artifacts_folder is missing, please run import (on pc)', artifacts_folder)
            raise FileNotFoundError(error_message)
        #
        # import may not be being done now - but artifacts folder exists,
        # we assume that it is proper and import is done
        self.is_imported = True
        self.is_start_infer_done = True

    def __call__(self, input, info_dict):
        return self.infer_frame(input, info_dict)

    def infer_frame(self, input, info_dict=None):
        assert self.is_imported, 'import_model() must be called before infer_frame()'
        if not self.is_start_infer_done:
            self.start_infer()
        #
        if info_dict is not None:
            info_dict['run_dir'] = self.get_param('run_dir')
        #
        # the over-ridden function in super class must return valid outputs
        return None, None

    def infer_frames(self, inputs, info_dict=None):
        outputs = []
        for input in inputs:
            output, info_dict = self.infer_frame(input, info_dict)
            outputs.append(output)
        #
        return outputs, info_dict

    def run(self, calib_data, inputs, info_dict=None):
        info_dict = self.import_model(calib_data, info_dict)
        outputs, info_dict = self.infer_frames(inputs, info_dict)
        infer_stats_dict = self.infer_stats()
        info_dict.update(infer_stats_dict)
        return outputs, info_dict

    def __del__(self):
        for t in self.tempfiles:
            t.cleanup()
        #

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
            stats_dict['perfsim_macs'] = 0.0
            stats_dict['perfsim_time'] = 0.0
            stats_dict['perfsim_ddr_transfer'] = 0.0
        #
        return stats_dict

    def get_session_name(self):
        session_name = self.kwargs['session_name']
        return session_name

    def get_session_short_name(self):
        session_name = self.get_session_name()
        return constants.SESSION_NAMES_DICT[session_name]

    def _tidl_infer_stats(self):
        assert self.is_imported is True, 'the given model must be an imported one.'
        benchmark_dict = self.interpreter.get_TI_benchmark_data()
        subgraph_time = copy_time = 0
        cp_in_time = cp_out_time = 0
        subgraphIds = []
        for stat in benchmark_dict.keys():
            if 'proc_start' in stat:
                if self.kwargs['session_name'] == 'onnxrt':
                    value = stat.split("ts:subgraph_")
                    value = value[1].split("_proc_start")
                    subgraphIds.append(value[0])
                else:
                    subgraphIds.append(int(re.sub("[^0-9]", "", stat)))
                #
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
            'write_total': write_total, 'read_total': read_total,
            'perfsim_macs': 0.0, 'perfsim_time': 0.0, 'perfsim_ddr_transfer': 0.0
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
        run_dir_candidate = self.kwargs.get('run_dir', None)
        if run_dir_candidate is not None:
            return run_dir_candidate
        #
        # MemoryTempfile() creates a file in RAM, which should be really fast.
        work_dir = self.kwargs['work_dir']
        if work_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
            work_dir = os.path.join(temp_dir.name, date)
            self.tempfiles.append(temp_dir)
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

    def get_model(self, meta_file_key='object_detection:meta_layers_names_list'):
        model_folder = self.kwargs['model_folder']

        # download the file if it is an http or https link
        model_path = self.kwargs['model_path']
        # make a local copy
        model_file = utils.get_local_path(model_path, model_folder)
        # self.kwargs['model_file'] is what is used in the session
        # we could have just used self.kwargs['model_path'], but do this for legacy reasons
        self.kwargs['model_file'] = model_file

        print(utils.log_color('INFO', 'model_path', model_path))
        print(utils.log_color('INFO', 'model_file', model_file))

        model_file0 = model_file[0] if isinstance(model_file, (list,tuple)) else model_file
        model_file_exists = utils.file_exists(model_file0)
        if not model_file_exists:
            model_path = utils.download_files(model_path, root=model_folder)
        #
        # optimize the model to speedup inference.
        # for example, the input of the model can be converted to 8bit and mean/scale can be moved inside the model
        if self.kwargs['input_optimization'] and self.kwargs['tensor_bits'] == 8 and \
                self.kwargs['input_mean'] is not None and self.kwargs['input_scale'] is not None:
            optimization_done = self._optimize_model(is_new_file=(not model_file_exists))
            if optimization_done:
                # set the mean and scale in kwargs to None as they have been absorbed inside.
                self.kwargs['input_mean'] = None
                self.kwargs['input_scale'] = None
            #
        #
        if self.kwargs['input_mean'] is not None and self.kwargs['input_scale'] is not None:
            # mean scale could not be absorbed inside the model - do it explicitly
            self.input_normalizer = ImageNormMeanScale(
                self.kwargs['input_mean'], self.kwargs['input_scale'],
                self.kwargs['input_data_layout'])
        #
        # meta_file
        meta_path = self.kwargs['runtime_options'].get(meta_file_key, None)
        if meta_path is not None:
            # make a local copy
            meta_file = utils.get_local_path(meta_path, model_folder)
            if not utils.file_exists(meta_file):
                meta_path = utils.download_file(meta_path, root=model_folder)
            #
            # write the local path
            self.kwargs['runtime_options'][meta_file_key] = meta_file
        #

    def _optimize_model(self, is_new_file=True):
        model_file = self.kwargs['model_file']
        model_file0 = model_file[0] if isinstance(model_file, (list,tuple)) else model_file
        input_mean = self.kwargs['input_mean']
        input_scale = self.kwargs['input_scale']
        optimization_done = False
        if model_file0.endswith('.onnx'):
            if is_new_file:
                from osrt_model_tools.onnx_tools import onnx_model_opt as onnxopt
                onnxopt.tidlOnnxModelOptimize(model_file0, model_file0, input_scale, input_mean)
            #
            optimization_done = True
        elif model_file0.endswith('.tflite'):
            if is_new_file:
                from osrt_model_tools.tflite_tools import tflite_model_opt as tflopt
                tflopt.tidlTfliteModelOptimize(model_file0, model_file0, input_scale, input_mean)
            #
            optimization_done = True
        #
        return optimization_done

    def _set_default_options(self):
        assert False, 'this function must be overridden in the derived class'

    def clear(self):
        # make sure that the artifacts_folder is cleanedup
        for root, dirs, files in os.walk(self.kwargs['artifacts_folder'], topdown=False):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]
        #

    def set_runtime_option(self, option, value):
        assert False, 'this function must be overridden'

    def get_runtime_option(self, option, default=None):
        assert False, 'this function must be overridden'

    def layer_info(self):
        # assert self.is_imported is True, 'the given model must be an imported one.'
        artifacts_folder = self.kwargs['artifacts_folder']
        subgraph_root = os.path.join(artifacts_folder, 'tempDir') \
            if os.path.isdir(os.path.join(artifacts_folder, 'tempDir')) else artifacts_folder
        layer_info_files = [os.path.join(subgraph_root, f) for f in os.listdir(subgraph_root)]
        layer_info_files = [f for f in layer_info_files if os.path.isfile(f) and f.endswith('layer_info.txt')]
        model_layer_info = []
        for subgraph_id, layer_info_file in enumerate(layer_info_files):
            subgraph_name = os.path.basename(layer_info_file).split('.')[0]
            perfinfo_dict = self._read_perf_info(os.path.join(subgraph_root, subgraph_name))
            subgraph_info = []
            with open(layer_info_file) as layer_info_fp:
                for layer_info_line_id, layer_info_line in enumerate(layer_info_fp):
                    layer_info_line = layer_info_line.rstrip().split(' ')
                    layer_id, data_id, layer_name = layer_info_line[0], layer_info_line[1], layer_info_line[2]
                    l_info = {'subgraph_name': subgraph_name, 'layer_id': layer_id, 'data_id': data_id, 'layer_name': layer_name}
                    if perfinfo_dict is not None:
                        l_info.update(perfinfo_dict[layer_info_line_id])
                    #
                    subgraph_info.append(l_info)
                #
            #
            model_layer_info.append(subgraph_info)
        #
        return model_layer_info

    def _read_perf_info(self, subgraph_path):
        format_line_dict = lambda d: {k.strip():v.strip() for k,v in d.items() if k is not None and v is not None}
        perfinfo_files = os.listdir(subgraph_path)
        perfinfo_csv_file = [f for f in perfinfo_files if f.endswith('.csv') and 'DSP' in f][0]
        with open(os.path.join(subgraph_path,perfinfo_csv_file)) as perfinfo_csv_fp:
            perfinfo_data = csv.DictReader(perfinfo_csv_fp, delimiter=',')
            perfinfo_dict = [format_line_dict(line_dict) for line_dict in perfinfo_data]
        #
        return perfinfo_dict

    #


if __name__ == '__main__':
    import_model = BaseRTSession()
