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


from .runtimes.presets import *


# pipeline type constants
PIPELINE_UNDEFINED = None
PIPELINE_ACCURACY = 'accuracy'
PIPELINE_GEN_CONFIG = 'gen_config'

# frequency of the core C7x/MMA processor that accelerates Deep Learning Tasks
# this constant is used to convert cycles to time : time = cycles / DSP_FREQ
DSP_FREQ = 1e9

# other common constants
MILLI_CONST = 1e3 # multiplication by 1000 is to convert seconds to milliseconds
MEGA_CONST = 1e6  # convert raw data to mega : example bytes to mega bytes (MB)
GIGA_CONST = 1e9
ULTRA_CONST = 1e6

# task_type
TASK_TYPE_CLASSIFICATION = 'classification'
TASK_TYPE_DETECTION = 'detection'
TASK_TYPE_SEGMENTATION = 'segmentation'
TASK_TYPE_KEYPOINT_DETECTION = 'keypoint_detection'
TASK_TYPE_DEPTH_ESTIMATION = 'depth_estimation'
TASK_TYPE_DETECTION_3DOD = 'detection_3d'
TASK_TYPE_OBJECT_6D_POSE_ESTIMATION = 'object_6d_pose_estimation'
TASK_TYPE_VISUAL_LOCALIZATION = 'visual_localization'

TASK_TYPES = [
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_DETECTION,
    TASK_TYPE_SEGMENTATION,
    TASK_TYPE_KEYPOINT_DETECTION,
    TASK_TYPE_DEPTH_ESTIMATION,
    TASK_TYPE_DETECTION_3DOD,
    TASK_TYPE_OBJECT_6D_POSE_ESTIMATION,
    TASK_TYPE_VISUAL_LOCALIZATION
]

# supported model types
MODEL_TYPE_ONNX = 'onnx'
MODEL_TYPE_TFLITE = 'tflite'
MODEL_TYPE_MXNET = 'mxnet'
MODEL_TYPES = [MODEL_TYPE_ONNX, MODEL_TYPE_TFLITE, MODEL_TYPE_MXNET]

# supported session names
SESSION_NAME_TVMDLR = 'tvmdlr'
SESSION_NAME_TFLITERT = 'tflitert'
SESSION_NAME_ONNXRT = 'onnxrt'
SESSION_NAMES = [SESSION_NAME_ONNXRT, SESSION_NAME_TFLITERT, SESSION_NAME_TVMDLR]
SESSION_NAMES_DICT = {SESSION_NAME_ONNXRT:'ONR', SESSION_NAME_TFLITERT:'TFL', SESSION_NAME_TVMDLR:'TVM'}


class QUANTScaleType:
    QUANT_SCALE_TYPE_NP2 = 0              # 0 (non-power of 2, default)
    QUANT_SCALE_TYPE_P2 = 1               # 1 (power of 2, might be helpful sometimes, needed for p2 qat models)
    QUANT_SCALE_TYPE_UNUSED = 2           # 3 (non-power of 2 qat/prequantized model, supported in newer devices)
    # these are not supported in TDA4VM, but for other SoCs, these are the recommended modes
    QUANT_SCALE_TYPE_PREQUANT_TFLITE = 3  # 4 (non-power2 of 2, supported in newer devices)
    # per-channel quantization is highy recommended if this feature is supported in hardware
    QUANT_SCALE_TYPE_NP2_PERCHAN = 4      # per-channel quantization - supported in SoCs other than TDA4VM


class PreQuantizedModelType:
    PREQUANTIZED_MODEL_TYPE_NONE = None
    PREQUANTIZED_MODEL_TYPE_CLIP = 0
    PREQUANTIZED_MODEL_TYPE_QDQ = 1


# some options in runtime_options
OBJECT_DETECTION_META_FILE_KEY = 'object_detection:meta_layers_names_list'
ADVANCED_OPTIONS_QUANT_FILE_KEY = 'advanced_options:quant_params_proto_path'


# errors emitted in the log file to help identify a FATAL error
# look for erros such as: VX_ZONE_ERROR:[tivxDeInit:245] De-Initialization Error !!!
# but do not include genuine messages such as: VX_ZONE_ERROR:Enabled
TIDL_FATAL_ERROR_LOGS_REGEX_LIST = [r'VX_ZONE_ERROR:\[.*\].*Error.*']


# Compatible with the publicly released SDK 10.1 - Version: 10.01.00.04 - Release date: Dec 18, 2024
TIDL_FIRMWARE_VERSION_10_01_03_00 = '10_01_03_00'
TIDL_FIRMWARE_VERSION_SDK_DEFAULT = TIDL_FIRMWARE_VERSION_10_01_03_00

# Not automatically enabled and also not compatible with publicly released version of SDK 10.1 - needs firmware update in SDK.
TIDL_FIRMWARE_VERSION_10_01_04_00 = '10_01_04_00'
TIDL_FIRMWARE_VERSION_TIDL_TOOLS_LATEST_BUGFIX = TIDL_FIRMWARE_VERSION_10_01_04_00
