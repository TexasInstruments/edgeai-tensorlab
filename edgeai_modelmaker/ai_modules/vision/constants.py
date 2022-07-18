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

# task_type
TASK_TYPE_CLASSIFICATION = 'classification'
TASK_TYPE_DETECTION = 'detection'

TASK_TYPES = [
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_DETECTION
]

# target_device
TARGET_DEVICE_TDA4VM = 'TDA4VM'

TARGET_DEVICES = [
    TARGET_DEVICE_TDA4VM,
]

# training_device
TRAINING_DEVICE_CPU = 'cpu'
TRAINING_DEVICE_CUDA = 'cuda'
TRAINING_DEVICE_GPU = TRAINING_DEVICE_CUDA

TRAINING_DEVICES = [
    TRAINING_DEVICE_CPU,
    TRAINING_DEVICE_CUDA
]

TARGET_DEVICE_DESCRIPTIONS = {
    TARGET_DEVICE_TDA4VM : \
'''
* C7x floating point, vector DSP, up to 1.0 GHz, 80
GFLOPS, 256 GOPS
* Deep-learning matrix multiply accelerator (MMA),
up to 8 TOPS (8b) at 1.0 GHz
* Vision Processing Accelerators (VPAC) with Image
Signal Processor (ISP) and multiple vision assist
accelerators
* Depth and Motion Processing Accelerators
(DMPAC)
* Dual 64-bit Arm® Cortex®-A72 microprocessor
subsystem at up to 2.0 GHz
– 1MB shared L2 cache per dual-core Cortex®-
A72 cluster
– 32KB L1 DCache and 48KB L1 ICache per
Cortex®-A72 core
* Six Arm® Cortex®-R5F MCUs at up to 1.0 GHz
– 16K I-Cache, 16K D-Cache, 64K L2 TCM
– Two Arm® Cortex®-R5F MCUs in isolated MCU
subsystem
– Four Arm® Cortex®-R5F MCUs in general
compute partition
* Two C66x floating point DSP, up to 1.35 GHz,
40 GFLOPS, 160 GOPS
* 3D GPU PowerVR® Rogue 8XE GE8430, up to
750 MHz, 96 GFLOPS, 6 Gpix/sec
* Custom-designed interconnect fabric supporting
near max processing entitlement    
* For more information, see the following links:
- product page: https://www.ti.com/product/TDA4VM
- datasheet: https://www.ti.com/lit/ds/symlink/tda4vm.pdf
- starterkit: https://www.ti.com/tool/SK-TDA4VM
- top level landing page: https://ti.com/edgeai
'''
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
