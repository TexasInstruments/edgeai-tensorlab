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
import yaml
from . import utils


class ConfigDict(dict):
    def __init__(self, input=None, **kwargs):
        super().__init__()
        # initialize with default values
        self._initialize()
        # read the given settings file
        input_dict = dict()
        input_dict['settings_file'] = None
        if isinstance(input, str):
            ext = os.path.splitext(input)[1]
            assert ext == '.yaml', f'unrecognized file type for: {input}'
            with open(input) as fp:
                input_dict = yaml.safe_load(fp)
            #
            input_dict['settings_file'] = input
        elif isinstance(input, dict):
            input_dict = input
        elif input is not None:
            assert False, 'got invalid input'
        #
        # override the entries with kwargs
        for k, v in kwargs.items():
            input_dict[k] = v
        #
        for key, value in input_dict.items():
            if key == 'include_files':
                settings_file = input_dict['settings_file']
                include_base_path = os.path.dirname(settings_file) if settings_file is not None else './'
                idict = self._parse_include_files(value, include_base_path)
                self.update(idict)
            else:
                self.__setattr__(key, value)
            #
        #

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    # pickling used by multiprocessing did not work without defining __getstate__
    def __getstate__(self):
        self.__dict__.copy()

    # this seems to be not required by multiprocessing
    def __setstate__(self, state):
        self.__dict__.update(state)

    def _initialize(self):
        # include additional files and merge with this dict
        self.include_files = None
        # execution pipeline type - currently only accuracy pipeline is defined
        self.pipeline_type = 'accuracy'
        # number of frames for inference
        self.num_frames = 10000 #50000
        # number of frames to be used for post training quantization / calibration
        self.max_frames_calib = 50 #100
        # number of itrations to be used for post training quantization / calibration
        self.max_calib_iterations = 50
        # clone the modelzoo repo and make sure this folder is available.
        self.modelzoo_path = '../jacinto-ai-modelzoo/models'
        # create your datasets under this folder
        self.datasets_path = f'./dependencies/datasets'
        # important parameter. set this to 'pc' to do import and inference in pc
        # set this to 'j7' to run inference in device. for inference on device run_import
        # below should be switched off and it is assumed that the artifacts are already created.
        self.target_device = 'pc' #'j7' #'pc'
        # for parallel execution on pc (cpu or gpu). if you don't have gpu, these actual numbers don't matter,
        # but the size of teh list determines the number of parallel processes
        # if you have gpu's these entries can be gpu ids which will be used to set CUDA_VISIBLE_DEVICES
        self.parallel_devices = None #[0,1,2,3,0,1,2,3]
        # quantization bit precision
        self.tidl_tensor_bits = 8 #8 #16 #32
        # run import of the model - only to be used in pc - set this to False for j7 evm
        # for pc this can be True or False
        self.run_import = True
        # run inference - for inferene in j7 evm, it is assumed that the artifaacts folders are already available
        self.run_inference = True #True
        # collect final accuracy results
        self.collect_results = True #True
        # detection threshold
        self.detection_thr = 0.05
        # max number of detections
        self.detection_max = None
        # save detection, segmentation output
        self.save_output = False
        # wild card list to match against the model_path - if null, all models wil be run
        # examples: ['classification'] ['imagenet1k'] ['torchvision']
        # examples: ['resnet18_opset9.onnx', 'resnet50_v1.tflite']
        self.model_selection = None
        # exclude the models that matches with this
        self.model_exclusion = None
        # wild card list to match against the tasks. it null, all tasks will be run
        # example: ['classification', 'detection', 'segmentation']
        # example: ['classification']
        self.task_selection = None
        # session types to use for each model type
        self.session_type_dict = {'onnx': 'tvmdlr', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'}
        # whether to load the datasets or not. set to False to load no datasets
        # set to True to try and load all datasets (the dataset folders must be available in ./dependencies/datasets).
        # for selective loading, provide a list of dataset names such as ['imagenet', 'coco', 'cityscapes', 'ade20k', 'voc2012']
        self.dataset_loading = True
        # which configs to run from the default list. example [0,10] [10,null] etc.
        self.config_range = None
        # logging of the import, infer and the accuracy. set to False to disable it.
        self.enable_logging = True
        # verbose mode - print out more information
        self.verbose = False
        # package the artifacts
        self.package_artifacts = False

        ###########################################################
        # internal state information
        ###########################################################
        self.dataset_cache = None

    def _parse_include_files(self, include_files, include_base_path):
        input_dict = {}
        include_files = utils.as_list(include_files)
        for include_file in include_files:
            append_base = not (include_file.startswith('/') and include_file.startswith('./'))
            include_file = os.path.join(include_base_path, include_file) if append_base else include_file
            with open(include_file) as ifp:
                idict = yaml.safe_load(ifp)
                input_dict.update(idict)
            #
        #
        return input_dict

