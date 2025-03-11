#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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

import torch
from .normalization import *

# optional/experimental
try: from .conv_ws_internal import *
except: pass


###############################################################
#class NoTrackBatchNorm2d(torch.nn.BatchNorm2d):
#    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
#        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
#track_running_stats=False)


###############################################################
# Default Normalization used
# DefaultNorm2d can be set to one of the normalization types
###############################################################
DefaultNorm2d = torch.nn.BatchNorm2d #SlowBatchNorm2d #Group8Norm


###############################################################
# Default Activation
# DefaultAct2d can be set to one of the activation types
###############################################################
DefaultAct2d = torch.nn.ReLU #torch.nn.HardTanh

###############################################################
# Default Convolution: torch.nn.Conv2d or ConvWS2d
###############################################################
DefaultConv2d = torch.nn.Conv2d