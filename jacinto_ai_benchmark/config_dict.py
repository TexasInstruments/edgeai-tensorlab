import os
import yaml


class ConfigDict(dict):
    def __init__(self, input=None, **kwargs):
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
        super().__init__(input_dict)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def _initialize(self):
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
        # save detection, segmentation output
        self.save_output = False
        # wild card list to match against the model_path - if null, all models wil be run
        # examples: ['classification'] ['imagenet1k'] ['torchvision']
        # examples: ['resnet18_opset9.onnx', 'resnet50_v1.tflite']
        self.model_selection = None
        # wild card list to match against the tasks. it null, all tasks will be run
        # example: ['classification', 'detection', 'segmentation']
        # example: ['classification']
        self.task_selection = None
        # whether to load the datasets or not
        self.dataset_loading = True
        # which configs to run from the default list. example [0,10] [10,null] etc.
        self.config_range = None
        # verbose mode - print out more information
        self.verbose = False
