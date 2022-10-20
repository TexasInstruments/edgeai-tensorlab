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

# plugin types
PLUGINS_ENABLE_GPL = True

# task_type
TASK_TYPE_CLASSIFICATION = 'classification'
TASK_TYPE_DETECTION = 'detection'

TASK_TYPES = [
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_DETECTION
]

# target_device
TARGET_DEVICE_TDA4VM = 'TDA4VM'
TARGET_DEVICE_AM62 = 'AM62'

TARGET_DEVICES = [
    TARGET_DEVICE_TDA4VM,
    TARGET_DEVICE_AM62
]

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

TARGET_DEVICE_DETAILS_TDA4VM = '''
The TDA4VM processor family is based on the evolutionary Jacinto™ 7 architecture, targeted at ADAS, 
Autonomous Vehicle (AV) and industrial applications applications. 
The unique combination high-performance compute, deeplearning engine, 
dedicated accelerators for signal and image processing in an functional safety compliant
targeted architecture make the TDA4VM devices a great fit for several industrial applications, such as: 
Robotics, Machine Vision, Radar, and so on. 
Key cores include next generation DSP with scalar and vector cores,
dedicated deep learning and traditional algorithm accelerators, latest Arm and GPU processors for general
compute, an integrated next generation imaging subsystem (ISP), video codec, Ethernet hub and isolated MCU
island. All protected by automotive grade safety and security hardware accelerators.
Processor Cores:
* C7x floating point, vector DSP, up to 1.0 GHz, 80 GFLOPS, 256 GOPS
* Deep-learning matrix multiply accelerator (MMA), up to 8 TOPS (8b) at 1.0 GHz
* Vision Processing Accelerators (VPAC) with Image Signal Processor (ISP) and multiple vision assist accelerators
* Depth and Motion Processing Accelerators (DMPAC)
* Dual 64-bit Arm® Cortex®-A72 microprocessor subsystem at up to 2.0 GHz
* Six Arm® Cortex®-R5F MCUs at up to 1.0 GHz compute partition
* Two C66x floating point DSP, up to 1.35 GHz, 40 GFLOPS, 160 GOPS
* 3D GPU PowerVR® Rogue 8XE GE8430, up to 750 MHz, 96 GFLOPS, 6 Gpix/sec
* Custom-designed interconnect fabric supporting near max processing entitlement    
For more information, see the following links:
- product page: https://www.ti.com/product/TDA4VM
- starterkit: https://www.ti.com/tool/SK-TDA4VM
- top level landing page: https://ti.com/edgeai
- software development page - introduction: https://dev.ti.com/edgeai/
- software development page - model development: https://github.com/TexasInstruments/edgeai
- software development page - sdk: https://www.ti.com/tool/PROCESSOR-SDK-J721E
'''

TARGET_DEVICE_DETAILS_AM62 = '''
The low-cost AM62x Sitara™ MPU family of application processors are built for Linux® application development.
With scalable Arm® Cortex®-A53 performance and embedded features, such as: dual-display support and 3D
graphics acceleration, along with an extensive set of peripherals that make the AM62x device well-suited for
a broad range of industrial and automotive applications while offering intelligent features and optimized power architecture as well.
Processor Cores:
* Up to Quad 64-bit Arm® Cortex®-A53 microprocessor subsystem at up to 1.4 GHz
* Single-core Arm® Cortex®-M4F MCU at up to 400 MHz
* Dedicated Device and Power Manager
Multimedia:
* Display subsystem
* 3D Graphics Processing Unit
* One Camera Serial interface (CSI-Rx) - 4 Lane
with DPHY
For more information, see the following links:
- product page: https://www.ti.com/product/AM623
- product page: https://www.ti.com/product/AM625
- starterkit: https://www.ti.com/tool/SK-AM62
- top level landing page: https://ti.com/edgeai
- software development page - introduction: https://dev.ti.com/edgeai/
- software development page - model development: https://github.com/TexasInstruments/edgeai
- software development page - sdk: https://www.ti.com/tool/PROCESSOR-SDK-AM62X
'''

TARGET_DEVICE_DESCRIPTIONS = {
    TARGET_DEVICE_TDA4VM : {
        'device_name': TARGET_DEVICE_TDA4VM,
        'device_type': TARGET_DEVICE_TYPE_MPU,
        'device_selection_factor': 1,
        'device_details' : TARGET_DEVICE_DETAILS_TDA4VM,
    },
    TARGET_DEVICE_AM62 : {
        'device_name': TARGET_DEVICE_AM62,
        'device_type': TARGET_DEVICE_TYPE_MPU,
        'device_selection_factor': 0,
        'device_details' : TARGET_DEVICE_DETAILS_AM62,
    }
}

TASK_DESCRIPTIONS = {
    TASK_TYPE_CLASSIFICATION: {
        'target_module': 'vision',
        'target_devices': [TARGET_DEVICE_TDA4VM, TARGET_DEVICE_AM62],
        'stages': ['dataset', 'training', 'compilation'],
    },
    TASK_TYPE_DETECTION: {
        'target_module': 'vision',
        'target_devices': [TARGET_DEVICE_TDA4VM, TARGET_DEVICE_AM62],
        'stages': ['dataset', 'training', 'compilation'],
    }
}

PRESET_DESCRIPTIONS = {
    TARGET_DEVICE_TDA4VM: {
        'best_accuracy_preset': dict(
            compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.05, tensor_bits=8)
        ),
        'high_accuracy_preset': dict(
            compilation=dict(calibration_frames=25, calibration_iterations=25, detection_threshold=0.3, tensor_bits=8)
        ),
        'default_preset': None,  # not specified here - use the models values
        'high_speed_preset': dict(
            compilation=dict(calibration_frames=5, calibration_iterations=5, detection_threshold=0.3, tensor_bits=8)
        ),
        'best_speed_preset': dict(
            compilation=dict(calibration_frames=5, calibration_iterations=1, detection_threshold=0.3, tensor_bits=8)
        ),
    },
    TARGET_DEVICE_AM62: {
        'best_accuracy_preset': dict(
            compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.05, tensor_bits=32, tidl_offload=False)
        ),
        'high_accuracy_preset': dict(
            compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.3, tensor_bits=32, tidl_offload=False)
        ),
        'default_preset': dict(
            compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.3, tensor_bits=32, tidl_offload=False)
        ),
        'high_speed_preset': dict(
            compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.3, tensor_bits=32, tidl_offload=False)
        ),
        'best_speed_preset': dict(
            compilation=dict(calibration_frames=1, calibration_iterations=1, detection_threshold=0.3, tensor_bits=32, tidl_offload=False)
        ),
    },
}

SAMPLE_DATASET_DESCRIPTIONS = {
    'animal_classification': {
        'task_type': TASK_TYPE_CLASSIFICATION,
        'download_path': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/datasets/animal_classification.zip'
    },
    'tiscapes2017_driving' : {
        'task_type': TASK_TYPE_DETECTION,
        'download_path': 'http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/datasets/tiscapes2017_driving.zip'
    }
}

