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

###############################################################
# SlowBatchNorm2d is basically same as BatchNorm2d, but with a slower update of batch norm statistics.
# When the statistics are updated slowly, they tend to be similar across GPUs.
# This may help to minimize the problem of lower of inference accuracy when training with smaller batch sizes.
class SlowBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.01, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)


###############################################################
# Experimental - group norm with fixed number of channels
class Group4Norm(torch.nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(num_groups=4, num_channels=num_channels)


class Group8Norm(torch.nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(num_groups=8, num_channels=num_channels)


###############################################################
# Experimental - a trick to increase the batch size in batch norm by using lower number of channels internally
# This may help to minimize the problem of lower of inference accuracy when training with smaller batch sizes.
class GroupBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_groups, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        assert num_features % num_groups == 0, 'num_features={} is not divisible by num_groups={}'.format(num_features, num_groups)
        super().__init__(num_features//num_groups, eps, momentum, affine, track_running_stats)
        self.num_groups = num_groups


    def forward(self, x):
        b,c,h,w = x.size()
        x_grouped = x.view(-1,c//self.num_groups,h,w).contiguous()
        y_gropued = super().forward(x_grouped)
        y = y_gropued.view(b,c,h,w).contiguous()
        return y


class Group4BatchNorm2d(GroupBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(4, num_features, eps, momentum, affine, track_running_stats)


class Group8BatchNorm2d(GroupBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(8, num_features, eps, momentum, affine, track_running_stats)

