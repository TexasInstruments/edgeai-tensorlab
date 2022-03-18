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
TIDL_VERSION = (8,2)
TIDL_VERSION_STR = '8.2'

# data layout constants
NCHW = 'NCHW'
NHWC = 'NHWC'

# pipeline type constants
PIPELINE_UNDEFINED = None
PIPELINE_ACCURACY = 'accuracy'
PIPELINE_COMPARE = 'compare'

# frequency of the core C7x/MMA processor that accelerates Deep Learning Tasks
# this constant is used to convert cycles to time : time = cycles / DSP_FREQ
DSP_FREQ = 1e9

# other common constants
MILLI_CONST = 1e3 # multiplication by 1000 is to convert seconds to milliseconds
MEGA_CONST = 1e6  # convert raw data to mega : example bytes to mega bytes (MB)
GIGA_CONST = 1e9
ULTRA_CONST = 1e6

# supported model types
MODEL_TYPE_ONNX = 'onnx'
MODEL_TYPE_TFLITE = 'tflite'
MODEL_TYPE_MXNET = 'mxnet'
MODEL_TYPES = [MODEL_TYPE_ONNX, MODEL_TYPE_TFLITE, MODEL_TYPE_MXNET]

# supported session names
SESSION_NAME_TVMDLR = 'tvmdlr'
SESSION_NAME_TFLITERT = 'tflitert'
SESSION_NAME_ONNXRT = 'onnxrt'
SESSION_NAMES = [SESSION_NAME_TVMDLR, SESSION_NAME_TFLITERT, SESSION_NAME_ONNXRT]
SESSION_NAMES_DICT = {SESSION_NAME_TVMDLR:'TVM', SESSION_NAME_TFLITERT:'TFL', SESSION_NAME_ONNXRT:'ONR'}
