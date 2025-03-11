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


# TIDL platform to be used for compilation
# This also depends on the tidl_tools that is installed.
TIDL_PLATFORM = "J7"


# TIDL version that is supported by default - however this is not the only version that is supported.
# This version actually depends on tidl_tools that is being used - so what is populated here is just for guidance.
TIDL_VERSION = (10,1)
TIDL_VERSION_STR = '10.1'


# target devices/socs supported.
TARGET_DEVICE_TDA4VM = 'TDA4VM'
TARGET_DEVICE_AM62A = 'AM62A'
TARGET_DEVICE_AM67A = 'AM67A'
TARGET_DEVICE_AM68A = 'AM68A'
TARGET_DEVICE_AM69A = 'AM69A'
TARGET_DEVICE_AM62 = 'AM62'


# compilation can only be run in PC as of now, but inference can be run in both PC and EVM
# whether running in PC/Host Emulation or really running in EVM/device:
TARGET_MACHINE_PC_EMULATION = 'pc'
TARGET_MACHINE_EVM = 'evm'


# data layout constants
NCHW = 'NCHW'
NHWC = 'NHWC'


# supported model types
MODEL_TYPE_ONNX = 'onnx'
MODEL_TYPE_TFLITE = 'tflite'
MODEL_TYPE_MXNET = 'mxnet'
MODEL_TYPES = [MODEL_TYPE_ONNX, MODEL_TYPE_TFLITE, MODEL_TYPE_MXNET]


class QUANTScaleType:
    QUANT_SCALE_TYPE_NP2 = 0              # 0 (non-power of 2, default)
    QUANT_SCALE_TYPE_P2 = 1               # 1 (power of 2, might be helpful sometimes, needed for p2 qat models)
    QUANT_SCALE_TYPE_UNUSED = 2           # 3 (non-power of 2 qat/prequantized model, supported in newer devices)
    # these are not supported in TDA4VM, but for other SoCs, these are the recommended modes
    QUANT_SCALE_TYPE_PREQUANT_TFLITE = 3  # 4 (non-power2 of 2, supported in newer devices)
    # per-channel quantization is highy recommended if this feature is supported in hardware
    QUANT_SCALE_TYPE_NP2_PERCHAN = 4      # per-channel quantization - supported in SoCs other than TDA4VM


CALIBRATION_ITERATIONS_FACTOR_1X = 1.0
CALIBRATION_ITERATIONS_FACTOR_NX = 2.0


# runtime_options preferred - may not blindly apply for qat models
TARGET_DEVICE_SETTINGS_PRESETS = {
    TARGET_DEVICE_TDA4VM : {
        # TDA4VM does not support the per-channel asymmetric quantization - p2 can be used as default
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_P2},
        # we may need more calibration images and iterations
        'calibration_iterations_factor': CALIBRATION_ITERATIONS_FACTOR_NX
    },
    TARGET_DEVICE_AM62A : {
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN},
    },
    TARGET_DEVICE_AM67A: {
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN},
    },
    TARGET_DEVICE_AM68A : {
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN},
    },
    TARGET_DEVICE_AM69A : {
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN},
    },
    TARGET_DEVICE_AM62 : {
        'runtime_options': {},
        'tidl_offload': False,
    },
}


class PreQuantizedModelType:
    PREQUANTIZED_MODEL_TYPE_NONE = None
    PREQUANTIZED_MODEL_TYPE_CLIP = 0
    PREQUANTIZED_MODEL_TYPE_QDQ = 1


# to handle speciall case for runtime_options['object_detection:xx']
TIDL_DETECTION_META_ARCH_TYPE_SSD_TFLITE = 1
TIDL_DETECTION_META_ARCH_TYPE_SSD_ONNX = 3
TIDL_DETECTION_META_ARCH_TYPE_SSD_LIST = [
    TIDL_DETECTION_META_ARCH_TYPE_SSD_TFLITE,
    TIDL_DETECTION_META_ARCH_TYPE_SSD_ONNX,
]
