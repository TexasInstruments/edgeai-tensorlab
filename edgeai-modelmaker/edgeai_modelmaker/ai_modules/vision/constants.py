# Copyright (c) 2018-2021, Texas Instruments Incorporated
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


# plugins/additional models
# see the setup_all.sh file to understand how to set this
PLUGINS_ENABLE_EXTRA = False

# task_type
TASK_TYPE_CLASSIFICATION = 'classification'
TASK_TYPE_DETECTION = 'detection'
TASK_TYPE_SEGMENTATION = 'segmentation'
TASK_TYPE_KEYPOINT_DETECTION = 'keypoint_detection'

TASK_TYPES = [
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_DETECTION,
    TASK_TYPE_SEGMENTATION,
    #TASK_TYPE_KEYPOINT_DETECTION
]

# target_device
TARGET_DEVICE_TDA4VM = 'TDA4VM'
TARGET_DEVICE_AM62A = 'AM62A'
TARGET_DEVICE_AM68A = 'AM68A'
TARGET_DEVICE_AM69A = 'AM69A'
TARGET_DEVICE_AM62 = 'AM62'
TARGET_DEVICE_AM67A = 'AM67A'

TARGET_DEVICES = [
    TARGET_DEVICE_AM62,
    TARGET_DEVICE_AM62A,
    TARGET_DEVICE_AM67A,
    TARGET_DEVICE_TDA4VM,
    TARGET_DEVICE_AM68A,
    TARGET_DEVICE_AM69A,
]

# include additional devices that are not currently supported in release.
TARGET_DEVICES_ALL = TARGET_DEVICES

TARGET_DEVICE_TYPE_MPU = 'MPU'
TARGET_DEVICE_TYPE_MCU = 'MCU'

TARGET_DEVICE_TYPES = [
    TARGET_DEVICE_TYPE_MPU,
    TARGET_DEVICE_TYPE_MCU
]

# training_device
TRAINING_DEVICE_CPU = 'cpu'
TRAINING_DEVICE_CUDA = 'cuda'
TRAINING_DEVICE_GPU = TRAINING_DEVICE_CUDA

TRAINING_DEVICES = [
    TRAINING_DEVICE_CPU,
    TRAINING_DEVICE_CUDA
]

TRAINING_BATCH_SIZE_DEFAULT = {
    TASK_TYPE_CLASSIFICATION: 64,
    TASK_TYPE_DETECTION: 8,
    TASK_TYPE_SEGMENTATION: 16,
    TASK_TYPE_KEYPOINT_DETECTION: 8
}


TARGET_SDK_VERSION = '10.0.0'
TARGET_SDK_RELEASE = '10_00_00'


EDGEAI_TARGET_DEVICE_ADDITIONAL_INFORMATION = \
f'''* Edge AI introduction: https://ti.com/edgeai
* Edge AI model development information: https://github.com/TexasInstruments/edgeai
* Edge AI tools introduction: https://dev.ti.com/edgeai/
'''

##### TDA4VM ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_TDA4VM = \
f'''* Product information: https://www.ti.com/product/TDA4VM
* Development board: https://www.ti.com/tool/SK-TDA4VM
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM
* SDK release: {TARGET_SDK_RELEASE}'''


TARGET_DEVICE_DETAILS_TDA4VM = \
f'''Efficient 8 TOPS AI capability at edge
Specification:
* 8 TOPS Deep Learning accelerator
* Dual Arm® Cortex®-A72
* Integrated ISP
* More details : https://www.ti.com/product/TDA4VM

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_TDA4VM}

Additional information:
{EDGEAI_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''


##### AM62A ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_AM62A = \
f'''* Product information: https://www.ti.com/product/AM62A7
* Development board: https://www.ti.com/tool/SK-AM62A-LP
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM62A
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM62A
* SDK release: {TARGET_SDK_RELEASE}'''


TARGET_DEVICE_DETAILS_AM62A = \
f'''Efficient 2 TOPS AI capability at edge
Specification:
* 2 TOPS Deep Learning accelerator
* Quad Arm® Cortex®-A53
* Integrated ISP
* More details : https://www.ti.com/product/AM62A7

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_AM62A}

Additional information:
{EDGEAI_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''


##### AM67A ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_AM67A = \
f'''* Product information: https://www.ti.com/product/AM67A
* Development board: https://www.ti.com/tool/J722SXH01EVM
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM67A
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM67A
* SDK release: {TARGET_SDK_RELEASE}'''


TARGET_DEVICE_DETAILS_AM67A = \
f'''Efficient 4 TOPS AI capability at edge
Specification:
* 4 TOPS Deep Learning accelerator
* Quad Arm® Cortex®-A53
* Integrated ISP
* More details : https://www.ti.com/product/AM67A

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_AM67A}

Additional information:
{EDGEAI_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''


##### AM68A ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_AM68A = \
f'''* Product information: https://www.ti.com/product/AM68A
* Development board: https://www.ti.com/tool/SK-AM68
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM68A
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM68A
* SDK release: {TARGET_SDK_RELEASE}'''


TARGET_DEVICE_DETAILS_AM68A = \
f'''Efficient 8 TOPS AI capability at edge
Specification:
* 8 TOPS Deep Learning accelerator
* Dual Arm® Cortex®-A72
* Integrated ISP
* More details : https://www.ti.com/product/AM68A

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_AM68A}

Additional information:
{EDGEAI_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''


##### AM69A ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_AM69A = \
f'''* Product information: https://www.ti.com/product/AM69A
* Development board: https://www.ti.com/tool/SK-AM69
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM69A
* SDK documentation & board setup: See Edge AI documentation at https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM69A
* SDK release: {TARGET_SDK_RELEASE}'''


TARGET_DEVICE_DETAILS_AM69A = \
f'''Efficient 32 TOPS AI capability at edge
Specification:
* 32 TOPS Deep Learning accelerator
* 8 Arm® Cortex®-A72
* Integrated ISP
* More details : https://www.ti.com/product/AM69A

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_AM69A}

Additional information:
{EDGEAI_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''

##### AM62 ######
TARGET_DEVICE_SETUP_INSTRUCTIONS_AM62 = \
f'''* Product information: https://www.ti.com/product/AM625
* Development board: https://www.ti.com/tool/SK-AM62
* Edge AI Linux SDK: https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-AM62X
* SDK documentation & board setup: See analytics application and Edge AI documentation at https://www.ti.com/tool/SK-AM62#order-start-development
* SDK release: {TARGET_SDK_RELEASE}'''


TARGET_DEVICE_DETAILS_AM62 = \
f'''Human-machine-interaction SoC
Specification:
* Arm® Cortex®-A53-based edge AI and full-HD dual display
* Up to Quad Arm® Cortex®-A53 at up to 1.4 GHz
* 3D Graphics Processing Unit
MHz
* More details : https://www.ti.com/product/AM625

Important links:
{TARGET_DEVICE_SETUP_INSTRUCTIONS_AM62}

Additional information:
{EDGEAI_TARGET_DEVICE_ADDITIONAL_INFORMATION}'''


# higher device_selection_factor indicates higher performance device.
TARGET_DEVICE_DESCRIPTIONS = {
    TARGET_DEVICE_AM62: {
        'device_name': TARGET_DEVICE_AM62,
        'device_type': TARGET_DEVICE_TYPE_MPU,
        'device_selection_factor': 0,
        'device_details': TARGET_DEVICE_DETAILS_AM62,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_AM62A: {
        'device_name': TARGET_DEVICE_AM62A,
        'device_type': TARGET_DEVICE_TYPE_MPU,
        'device_selection_factor': 1,
        'device_details': TARGET_DEVICE_DETAILS_AM62A,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_AM67A: {
        'device_name': TARGET_DEVICE_AM67A,
        'device_type': TARGET_DEVICE_TYPE_MPU,
        'device_selection_factor': 1,
        'device_details': TARGET_DEVICE_DETAILS_AM67A,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_TDA4VM: {
        'device_name': TARGET_DEVICE_TDA4VM,
        'device_type': TARGET_DEVICE_TYPE_MPU,
        'device_selection_factor': 2,
        'device_details': TARGET_DEVICE_DETAILS_TDA4VM,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_AM68A: {
        'device_name': TARGET_DEVICE_AM68A,
        'device_type': TARGET_DEVICE_TYPE_MPU,
        'device_selection_factor': 3,
        'device_details': TARGET_DEVICE_DETAILS_AM68A,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
    TARGET_DEVICE_AM69A: {
        'device_name': TARGET_DEVICE_AM69A,
        'device_type': TARGET_DEVICE_TYPE_MPU,
        'device_selection_factor': 4,
        'device_details': TARGET_DEVICE_DETAILS_AM69A,
        'sdk_version': TARGET_SDK_VERSION,
        'sdk_release': TARGET_SDK_RELEASE,
    },
}

TASK_DESCRIPTIONS = {
    TASK_TYPE_CLASSIFICATION: {
        'task_name': 'Image Classification',
        'target_module': 'vision',
        'target_devices': TARGET_DEVICES,
        'stages': ['dataset', 'training', 'compilation'],
    },
    TASK_TYPE_DETECTION: {
        'task_name': 'Object Detection',
        'target_module': 'vision',
        'target_devices': TARGET_DEVICES,
        'stages': ['dataset', 'training', 'compilation'],
    },
    TASK_TYPE_SEGMENTATION: {
       'task_name': 'Semantic Segmentation',
       'target_module': 'vision',
       'target_devices': TARGET_DEVICES,
       'stages': ['dataset', 'training', 'compilation'],
    },
    TASK_TYPE_KEYPOINT_DETECTION: {
       'task_name': 'Keypoint Detection',
       'target_module': 'vision',
       'target_devices': TARGET_DEVICES,
       'stages': ['dataset', 'training', 'compilation'],
    }
}

# compilation settings for various speed and accuracy tradeoffs:
# detection_threshold & detection_top_k are written to the prototxt - inside edgeai-benchmark.
# prototxt is not used in AM62 - so those values does not have effect in AM62 - they are given just for completeness.
# if we really wan't to change the detections settings in AM62, we will have to modify the onnx file, but that's not easy.

PRESET_DESCRIPTIONS = {
    TARGET_DEVICE_TDA4VM: {
        TASK_TYPE_CLASSIFICATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
        },
        TASK_TYPE_SEGMENTATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_KEYPOINT_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
        },
    },
    TARGET_DEVICE_AM62A: {
        TASK_TYPE_CLASSIFICATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
        },
        TASK_TYPE_SEGMENTATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_KEYPOINT_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
        },
    },
    TARGET_DEVICE_AM67A: {
        TASK_TYPE_CLASSIFICATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05,
                                 detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6,
                                 detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6,
                                 detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6,
                                 detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6,
                                 detection_top_k=200, tensor_bits=8)
            ),
        },
        TASK_TYPE_SEGMENTATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_KEYPOINT_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05,
                                 detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6,
                                 detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6,
                                 detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6,
                                 detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6,
                                 detection_top_k=200, tensor_bits=8)
            ),
        },
    },
    TARGET_DEVICE_AM68A: {
        TASK_TYPE_CLASSIFICATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
        },
        TASK_TYPE_SEGMENTATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None,
                                 detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_KEYPOINT_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
        },
    },
    TARGET_DEVICE_AM69A: {
        TASK_TYPE_CLASSIFICATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
        },
        TASK_TYPE_SEGMENTATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=None, detection_top_k=None, tensor_bits=8)
            ),
        },
        TASK_TYPE_KEYPOINT_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=16)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=8)
            ),
        },
    },
    TARGET_DEVICE_AM62: {
        TASK_TYPE_CLASSIFICATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
        },
        TASK_TYPE_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.05, detection_top_k=500, tensor_bits=32, tidl_offload=False)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.6, detection_top_k=200, tensor_bits=32, tidl_offload=False)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.6, detection_top_k=200, tensor_bits=32, tidl_offload=False)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.6, detection_top_k=200, tensor_bits=32, tidl_offload=False)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.6, detection_top_k=200, tensor_bits=32, tidl_offload=False)
            ),
        },
        TASK_TYPE_SEGMENTATION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=None, detection_top_k=None, tensor_bits=32, tidl_offload=False)
            ),
        },
        TASK_TYPE_KEYPOINT_DETECTION: {
            'best_accuracy_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.05, detection_top_k=500, tensor_bits=32)
            ),
            'high_accuracy_preset': dict(
                compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.6, detection_top_k=200, tensor_bits=32)
            ),
            'default_preset': dict(
                compilation=dict(calibration_frames=10, calibration_iterations=10, detection_threshold=0.6, detection_top_k=200, tensor_bits=32)
            ),
            'high_speed_preset': dict(
                compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.6, detection_top_k=200, tensor_bits=32)
            ),
            'best_speed_preset': dict(
                compilation=dict(calibration_frames=3, calibration_iterations=3, detection_threshold=0.6, detection_top_k=200, tensor_bits=32)
            ),
        },
    },
}

# TODO: Need to add custom keypoint dataset as a SAMPLE_DATASET_DESCRIPTIONS
SAMPLE_DATASET_DESCRIPTIONS = {
    'animal_classification': {
        'common': {
            'task_type': TASK_TYPE_CLASSIFICATION,
        },
        'dataset': {
            'dataset_name': 'animal_classification',
            'input_data_path': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_classification.zip',
        },
        'info': {
            'dataset_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_classification.zip',
            'dataset_detailed_name': 'Animal classification',
            'dataset_description': 'Example cat-dog image classification dataset with 2 categories and 118 images',
            'dataset_size': 16137224,
            'dataset_frames': 118,
            'dataset_source': 'CC0 Public Domain Images from creativecommons.org, annotations by TI',
            'dataset_license': 'CC0 1.0 Universal Public Domain Dedication'
        }
    },
    'animal_detection': {
        'common': {
            'task_type': TASK_TYPE_DETECTION,
        },
        'dataset': {
            'dataset_name': 'animal_detection',
            'input_data_path': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_detection.zip',
        },
        'info': {
            'dataset_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/animal_detection.zip',
            'dataset_detailed_name': 'Animal detection',
            'dataset_description': 'Example cat-dog object detection dataset with 2 categories and 99 images',
            'dataset_size': 15290214,
            'dataset_frames': 99,
            'dataset_source': 'CC0 Public Domain Images from creativecommons.org, annotations by TI',
            'dataset_license': 'CC0 1.0 Universal Public Domain Dedication'
        }
    },
    'tiscapes2017_driving_detection' : {
        'common': {
            'task_type': TASK_TYPE_DETECTION,
        },
        'dataset': {
            'dataset_name': 'tiscapes2017_driving',
            'input_data_path': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/tiscapes2017_driving.zip',
        },
        'info': {
            'dataset_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/tiscapes2017_driving.zip',
            'dataset_detailed_name': 'TIScapes driving detection',
            'dataset_description': 'Example driving scenario object detection dataset with 4 categories and 2116 images',
            'dataset_size': 461038628,
            'dataset_frames': 2116,
            'dataset_source': 'Images & annotations from TI',
            'dataset_license': 'BSD 3-Clause'
        }
    },
    'tiscapes2017_driving_segmentation' : {
        'common': {
            'task_type': TASK_TYPE_SEGMENTATION,
        },
        'dataset': {
            'dataset_name': 'tiscapes2017_driving',
            'input_data_path': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/tiscapes2017_driving.zip',
        },
        'info': {
            'dataset_url': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/tiscapes2017_driving.zip',
            'dataset_detailed_name': 'TIScapes driving detection',
            'dataset_description': 'Example driving scenario object detection dataset with 4 categories and 2116 images',
            'dataset_size': 461038628,
            'dataset_frames': 2116,
            'dataset_source': 'Images & annotations from TI',
            'dataset_license': 'BSD 3-Clause'
        }
    },
}
