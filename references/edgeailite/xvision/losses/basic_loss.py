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

import torch
from .loss_utils import *

__all__ = [
    'BasicElementwiseLossModule', 'BasicNormLossModule', 'BasicSplitLossModule', 'MeanLossModule'
]


class BasicElementwiseLossModule(torch.nn.Module):
    '''
    This is the basic class used for elementwise functions.
    Unlike Norm functions these don't reduce the chennels dimension to size 1
    The handling of sparse is different from that of Norm Loss
    '''
    def __init__(self, sparse=False, error_fn=None, error_name=None, is_avg=False, min_elements=0, square_root=False):
        super().__init__()
        self.sparse = sparse
        self.error_fn = error_fn
        self.error_name = error_name
        self.is_avg = is_avg
        self.min_elements = min_elements
        self.square_root = square_root

    def forward(self, input_img, input_flow, target_flow):
        # input_flow, target_flow = utils.crop_alike(input_flow, target_flow)
        # invalid flow is defined with both flow coordinates to be exactly 0
        if self.sparse:
            target_abs_sum = torch.sum(torch.abs(target_flow), dim=1, keepdim=True)
            valid = (target_abs_sum != 0).expand_as(target_flow)
            input_flow = input_flow[valid]
            target_flow = target_flow[valid]

        error_flow = self.error_fn(input_flow, target_flow)
        error_val = error_flow.mean()
        # sqrt as in RMSE operation
        error_val = torch.sqrt(error_val) if self.square_root else error_val
        if error_flow.dim() > 0 and len(error_flow) < self.min_elements:
            # nan_tensor
            error_val = torch.tensor(1.0, device=input_flow.device) * float('nan')
        #
        return (error_val)
    def clear(self):
        return
    def info(self):
        return {'value':'error', 'name':self.error_name, 'is_avg':self.is_avg}
    @classmethod
    def args(cls):
        return ['sparse']


class BasicNormLossModule(torch.nn.Module):
    '''
    This is the basic class used for norm functions.
    Norm functions usually reduce the chennels dimension to size 1
        The handling of sparse is different from that of Elementiwise Loss
    '''
    def __init__(self, sparse=False, error_fn=None, error_name=None, is_avg=False):
        super().__init__()
        self.sparse = sparse
        self.error_fn = error_fn
        self.error_name = error_name
        self.is_avg = is_avg

    def forward(self, input_img, input_flow, target_flow):
        # input_flow, target_flow = utils.crop_alike(input_flow, target_flow)
        # invalid flow is defined with both flow coordinates to be exactly 0
        error_flow = self.error_fn(input_flow, target_flow)

        if self.sparse:
            target_abs_sum = torch.sum(torch.abs(target_flow), dim=1, keepdim=True)
            valid = (target_abs_sum != 0)
            error_flow = error_flow[valid]

        error_val = error_flow.mean()
        return (error_val)
    def clear(self):
        return
    def info(self):
        return {'value':'error', 'name':self.error_name, 'is_avg':self.is_avg}
    @classmethod
    def args(cls):
        return ['sparse']


class BasicSplitLossModule(torch.nn.Module):
    def __init__(self, sparse=False, error_fn=None, error_name=None, is_avg=False, channels=None, losses=None, weights=None):
        super().__init__()
        self.sparse = sparse
        self.error_fn = error_fn
        self.error_name = error_name
        self.is_avg = is_avg
        self.channels = channels
        self.losses = torch.nn.ModuleList(losses)
        self.weights = weights

    def forward(self, input_img, input_flow, target_flow):
        ch_start = 0
        for idx, (ch, wt) in enumerate(zip(self.channels, self.weights)):
            ch_end = ch_start + ch
            input_flow_split = input_flow[:,ch_start:ch_end,...]
            target_flow_split = target_flow[:,ch_start:ch_end,...]
            ch_start = ch_end
            if idx == 0:
                total_loss = self.losses[idx](input_img, input_flow_split, target_flow_split)*wt
            else:
                total_loss = total_loss + self.losses[idx](input_img, input_flow_split, target_flow_split)*wt

        return total_loss

    def clear(self):
        return
    def info(self):
        return {'value':'error', 'name':self.error_name, 'is_avg':self.is_avg}
    @classmethod
    def args(cls):
        return ['sparse']


#loss computed on the mean
class MeanLossModule(torch.nn.Module):
    def __init__(self, sparse=False, error_fn=None, error_name=None, is_avg=False):
        super().__init__()
        self.sparse = sparse
        self.error_fn = error_fn
        self.error_name = error_name
        self.is_avg = is_avg

    def forward(self, input_img, input_flow, target_flow):
        #input_flow, target_flow = utils.crop_alike(input_flow, target_flow)
        # invalid flow is defined with both flow coordinates to be exactly 0
        if self.sparse:
            #mask = (target_flow == 0)
            mask = (torch.sum(torch.abs(target_flow),dim=1,keepdim=True) == 0)
            mask = mask.expand_as(target_flow)

            valid = (mask == False)
            input_flow = input_flow[valid]
            target_flow = target_flow[valid]
        #
        input_mean = input_flow.mean()
        target_mean = target_flow.mean()
        error_val = self.error_fn(input_mean, target_mean)
        return (error_val)
    def clear(self):
        return
    def info(self):
        return {'value':'error', 'name':self.error_name, 'is_avg':self.is_avg}
    @classmethod
    def args(cls):
        return ['sparse']


