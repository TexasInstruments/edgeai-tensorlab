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


from . import core
from . import constants, datasets, sessions


class ConfigSettings(core.ConfigRuntimeOptions):
    def __init__(self, input, **kwargs):
        super().__init__(input, **kwargs)

        # collect basic keys that are added during initialization
        # only these will be copied during call to basic_settings()
        self.basic_keys = list(self.keys())

        for k, v in kwargs.items():
            if k not in self.basic_keys:
                raise RuntimeError(f"ERROR: invalid parameter given: {k}")
            #
        #

        # variable to preload datasets - so that it is not separately created for each config
        self.dataset_cache = datasets.initialize_datasets(self)

    def _initialize(self):
        # include additional files and merge with this dict
        self.include_files = None
        # execution pipeline type - currently only accuracy pipeline is defined
        self.pipeline_type = 'accuracy'
        # firmware version of SDK
        self.c7x_firmware_version = None
        # number of frames for inference
        self.num_frames = 1000 #10000 #50000
        # number of frames to be used for post training quantization / calibration
        self.calibration_frames = 25 #50
        # number of iterations to be used for post training quantization / calibration
        self.calibration_iterations = 25 #50
        # folder where benchmark configs are defined. this should be python importable
        self.configs_path = './configs'
        # folder where models are available
        self.models_path = '../edgeai-modelzoo/models'
        # path where compiled model artifacts are placed
        self.modelartifacts_path = './work_dirs/modelartifacts/{target_device}'
        # path where compiled packaged model artifacts are placed
        self.modelpackage_path = './work_dirs/modelpackage/{target_device}'
        # create your datasets under this folder
        self.datasets_path = f'./dependencies/datasets'
        # target_device indicates the SoC for which the model compilation will take place
        # see device_types for various devices in constants.TARGET_DEVICES_DICT
        # currently this field is for information only
        # the actual target device depends on the tidl_tools being used.
        self.target_device = None
        # important parameter. set this to 'pc' to do import and inference in pc
        # set this to 'evm' to run inference in device/soc. for inference on device run_import
        # below should be switched off and it is assumed that the artifacts are already created.
        self.target_machine = 'pc' #'evm' #'pc'
        # artifacts - suffix : attach this suffix to the run_dir where artifacts are created
        self.run_suffix = None
        # important note: for parallel execution on CUDA/gpu. requires CUDA compiled TIDL (tidl_tools) is required.
        # if you have gpu's, these wil be used for CUDA_VISIBLE_DEVICES. eg. specify 4 will use the gpus: 0,1,2,3
        # it can also be specified as a list with actual GPU ids, instead of an integer: [0,1,2,3]
        self.parallel_devices = None
        # for parallel execution on pc only (cpu or gpu).
        # number fo parallel processes to run.
        # for example 8 will mean 8 models will run in parallel
        # for example 1 will mean one model will run (but in a separae processs from that of the main process)
        # None will mean one process will run, in the same process as the main
        self.parallel_processes = None
        # quantization bit precision
        self.tensor_bits = 8 #8 #16 #32
        # runtime_options can be specified as a dict. eg {'accuracy_level': 0}
        self.runtime_options = {}
        # run import of the model - only to be used in pc - set this to False for evm
        # for pc this can be True or False
        self.run_import = True
        # run inference - for inferene in evm, it is assumed that the artifaacts folders are already available
        self.run_inference = True
        # run only models for which the results are missing. if this is False, all configs will be run
        self.run_incremental = True
        # detection threshold
        # recommend 0.3 for best fps, 0.05 for accuracy measurement - set to True to set this automatically
        self.detection_threshold = 0.3
        # detection  - top_k boxes that go into nms - set to True to set this automatically
        # (this is an intermediate set, not the final number of boxes that are kept)
        # # recommend 200 for best fps, 500 for accuracy measurement
        self.detection_top_k = 200
        # detection  - NMS threshold to be used for detection - set to True to set this automatically
        self.detection_nms_threshold = None
        # max number of final detections - set to True to set this automatically
        self.detection_keep_top_k = None
        # save detection, segmentation output
        self.save_output = False
        # number of frames for example output
        self.num_output_frames = 50 #None
        # wild card list to match against model_path, model_id or model_type - if null, all models wil be shortlisted
        # only models matching these criteria will be considered - even for model_selection
        #   examples: ['onnx'] ['tflite'] ['mxnet'] ['onnx', 'tflite']
        #   examples: ['resnet18.onnx', 'resnet50_v1.tflite'] ['classification'] ['imagenet1k'] ['torchvision'] ['coco']
        #   examples: [cl-0000, od-2020, ss-2580, cl-3090, cl-3520, od-5120, ss-5710, cl-6360, od-8050, od-8220, od-8420, ss-8610, kd-7060, 6dpose-7200]
        self.model_selection = None
        # model_shortlist can be a number, which indicates a predefined shortlist, and a fraction of the models will be selected
        # model_shortlist and model_selection are complimentary - they can be used together.
        # 0 means no models, 1 means 1 model, 15 means roughly 15% of models, 100 means all the models.
        #   examples: 1
        #   examples: 15
        #   examples: 30
        self.model_shortlist = None
        # exclude the models that matches with this
        self.model_exclusion = None
        # wild card list to match against the tasks. it null, all tasks will be run
        # example: ['classification', 'detection', 'segmentation', 'keypoint_detection', 'detection_3d']
        # example: ['classification']
        self.task_selection = None
        # wild card list to match against runtime name. if null, all runtimes will be considered
        # example: ['onnxrt', 'tflitert', 'tvmdlr']
        # example: ['onnxrt']
        self.runtime_selection = None
        # session types to use for each model type
        self.session_type_dict = {'onnx': 'onnxrt', 'tflite': 'tflitert', 'mxnet': 'tvmdlr'}
        # dataset type to use if there are multiple variants for each dataset
        # example: {'imagenet':'imagenetv1'}
        # example: {'imagenet':'imagenetv2c'}
        self.dataset_type_dict = None
        # dataset_selection can be a list of dataset categories for which the models will be run
        self.dataset_selection = None
        # whether to load the datasets or not. set to False or null to load no datasets
        # set to True to try and load all datasets (the dataset folders must be available in ./dependencies/datasets).
        # for selective loading, provide a list of dataset names such as
        # ['imagenet', 'coco', 'cocoseg21', 'ade20k', 'cocokpts', 'kitti_lidar_det', 'ti-robokit_semseg_zed1hd', 'ycbv']
        self.dataset_loading = True
        # which configs to run from the default list. example [0,10] [10,null] etc.
        self.config_range = None
        # writing of dataset.yaml, config.yaml, param.yaml and result.yaml depends on this flag
        self.write_results = True
        # verbose mode - print out more information
        self.verbose = False
        # log_file mode - capture and redirect details logs to log file
        self.log_file = True
        # enable use of additional models - the actual model files may be in a different modelzoo repo (for example edgeai-modelzoo-cl)
        self.additional_models = False
        # enable use of experimental models - the actual model files are not available in modelzoo
        self.experimental_models = False
        # rewrite results with latest params if the result exists
        self.rewrite_results = False
        # it defines if we want to use udp postprocessing in human pose estimation.
        # Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
        # Data Processing for Human Pose Estimation (CVPR 2020).
        self.with_udp = True
        # it will add horizontally flipped images in info_dict and run inference over the flipped image also
        self.flip_test = False
        # the transformations that needs to be applied to the model itself. Note: this is different from pre-processing transforms
        self.model_transformation_dict = None
        # include perfsim stats in the report or not
        self.report_perfsim = False
        # use TIDL offload to speedup inference
        self.tidl_offload = True
        # input optimization to improve FPS: False or None
        # None will cause the default value set in sessions.__init__ to be used.
        self.input_optimization = None
        # how many parent folders to be included from the model path, while creating the run_dir
        # default value is defined in basert_session.py
        self.run_dir_tree_depth = None
        # wehther to apply predifened presets based on target_device
        self.target_device_preset = True
        # some models can use faster calibration (fewer frames and iterations)
        self.calibration_iterations_factor = None
        # timeout while running the benchmark - per model timeout
        self.instance_timeout = None
        # timeout while running the benchmark - overall timeout
        self.overall_timeout = None
        # sort model configs based on model_shortlist parameter provided for each model
        self.sort_pipeline_configs = True
        # check_errors
        self.check_errors = True
        # can use this file as template to cleanup model config file
        self.param_template_file = None

    def basic_settings(self):
        '''this only returns the basic settings.
        sometimes, there is no need to copy the entire settings
        which includes the dataset_cache'''
        return core.ConfigDict({k:self[k] for k in self.basic_keys})

    def get_session_name(self, model_type_or_session_name):
        assert model_type_or_session_name in constants.MODEL_TYPES + constants.SESSION_NAMES, \
            f'get_session_cfg: input must be one of model_types: {constants.MODEL_TYPES} ' \
            f'or session_names: {constants.SESSION_NAMES}'
        if model_type_or_session_name in constants.MODEL_TYPES:
            model_type = model_type_or_session_name
            session_name = self.session_type_dict[model_type]
        else:
            session_name = model_type_or_session_name
        #
        assert session_name in constants.SESSION_NAMES, \
            f'get_session_cfg: invalid session_name: {session_name}'
        return session_name

    def get_session_type(self, model_type_or_session_name):
        session_name = self.get_session_name(model_type_or_session_name)
        return sessions.get_session_name_to_type_dict()[session_name]


class CustomConfigSettings(ConfigSettings):
    def __init__(self, input, dataset_loading=False, **kwargs):
        super().__init__(input, dataset_loading=dataset_loading, **kwargs)
