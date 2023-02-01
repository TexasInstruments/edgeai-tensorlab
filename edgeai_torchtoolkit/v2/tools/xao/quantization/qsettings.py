#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

# there is an onnx export issue if we use torch.qint8 dtype
# so use toch.quint8 even for the symmetric case
# accuracy is poor when setting this to false - suspect some bugs in torch quantization
USE_INT8_DTYPE_FOR_SYMMETRIC = False #True #False

# COMPUTE_ACCURATE_QPARAMS enables accurate computation of scale and zero_point in Power2 mode.
# if this set, we use custom scale and zero_point computation code
COMPUTE_ACCURATE_QPARAMS = False #True

# In addition USE_FULL_RANGE_FOR_SYMMETRIC enables better utilization of range in symmetric mode
# calculate_qparams in the base class does not handle unsigned tensors in symmetric case correctly
# ideally unsigned tensors (such as the output of ReLU) can be quantized with 256 levels (0-255),
# but calculate_qparams in original torch observers uses 127 levels only in symmetric mode:
#     (0to127 out ot -128to127 if qscheme is torch.int8)
#     (128to255 out ot -128to127 if qscheme is torch.uint8)
# however, enabling this flag changes the behaviour in the Power2 observers
# which are used in our qconfig.get_basic_qat_qconfig()
# This has effect only if COMPUTE_ACCURATE_QPARAMS is set.
USE_FULL_RANGE_FOR_SYMMETRIC = False #True

# the defaults are fine, but can be modified here if needed.
UINT8_DTYPE_MIN_VALUE = 0
UINT8_DTYPE_MAX_VALUE = 255
INT8_DTYPE_MIN_VALUE = -128 #-127
INT8_DTYPE_MAX_VALUE = 127

# default value for USE_HISTOGRAM_OBSERVER
# setting to False since Histogram observer is quite slow
# this can be overriden while calling the get_qconfig_xx()
USE_HISTOGRAM_OBSERVER_DEFAULT = False #True

# change dtype based on the signedness of the tensor
# to use zero_point as 0 for symmetric mode.
USE_ADAPTIVE_DTYPE = True