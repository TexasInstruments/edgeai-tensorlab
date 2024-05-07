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

import os
import copy
import torch
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization import FakeQuantize
import statistics
import functools

from .... import xnn

from . import observer_types
from . import fake_quanitze_types
from . import qconfig_types


def adjust_gradual_quantization(self):
    '''
    adjust quantization parameters on epoch basis
    '''
    if self.__quant_params__.qconfig_mode != qconfig_types.QConfigMode.DEFAULT and self.__quant_params__.total_epochs >= 10:
        # find unstable layers and freeze them
        self.adaptive_freeze_layers(fake_quanitze_types.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES)


def is_fake_quant_with_param(self, pmodule, cmodule, fake_quant_types):
    num_params = len(list(pmodule.parameters(recurse=False)))
    return isinstance(cmodule, fake_quant_types) and num_params > 0


def adaptive_freeze_layers(self, fake_quant_types, **kwargs):
    epoch_gradual_quant_start = max(self.__quant_params__.total_epochs // 2, 1)
    if self.__quant_params__.qconfig_mode == qconfig_types.QConfigMode.FREEZE_DEPTHWISE_LAYERS:
        num_total_layers = 0
        self.__quant_params__.forzen_layer_names_list = []
        is_freezing_epoch = (self.__quant_params__.num_epochs_tracked >= epoch_gradual_quant_start)
        for pname, pmodule in list(self.named_modules()):
            is_input_conv_module = False
            is_depthwise_conv_module = False
            if isinstance(pmodule, torch.nn.Conv2d) and pmodule.in_channels < 8:
                # too less input channels, could be first conv module
                is_input_conv_module = True
            if isinstance(pmodule, torch.nn.Conv2d) and pmodule.groups == pmodule.in_channels:
                is_depthwise_conv_module = True
            #
            for cname, cmodule in list(pmodule.named_children()):
                if self.__quant_params__.is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                    is_frozen_layer = (is_input_conv_module or is_depthwise_conv_module)
                    if is_freezing_epoch and is_frozen_layer:
                        # stop updating quantization ranges and stats
                        pmodule.apply(torch.ao.quantization.disable_observer)
                        pmodule.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                        # stop updating parmeters
                        for param in pmodule.parameters(recurse=False):
                            param.requires_update = False
                        #
                        self.__quant_params__.forzen_layer_names_list.append(pname)
                    #
                    num_total_layers += 1
                #
            #
        #
        print(f"using adaptive quantization - qconfig_mode:{self.__quant_params__.qconfig_mode} "
              f"num_frozen_layers:{len(self.__quant_params__.forzen_layer_names_list)}/{num_total_layers} ")
    elif self.__quant_params__.qconfig_mode == qconfig_types.QConfigMode.FREEZE_UNSTABLE_LAYERS:
        num_total_layers = 0
        delta_change_list = []
        for pname, pmodule in list(self.named_modules()):
            for cname, cmodule in list(pmodule.named_children()):
                if is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                    cmodule.set_adaptive_params(detect_change=True, **kwargs)
                    delta_change_list.append(cmodule.delta_change)
                #
            #
        #
        if self.__quant_params__.num_epochs_tracked >= epoch_gradual_quant_start:
            is_freezing_start_epoch = (self.__quant_params__.num_epochs_tracked == epoch_gradual_quant_start)
            # find sign_change_threshold
            freeze_fraction = 0.15
            delta_change_min = 0.04
            topk_index = int((len(delta_change_list) - 1) * (1 - freeze_fraction))
            delta_change_knee = sorted(delta_change_list)[topk_index]
            delta_change_threshold = max(delta_change_knee, delta_change_min)

            # freeze layers with high sign change
            num_total_layers = 0
            for pname, pmodule in list(self.named_modules()):
                max_delta_change = 0.0
                for cname, cmodule in list(pmodule.named_children()):
                    if is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                        # once frozen, always frozen
                        is_frozen_layer = (pname in self.__quant_params__.forzen_layer_names_list)
                        is_high_change = is_freezing_start_epoch and (cmodule.delta_change >= delta_change_threshold)
                        if is_frozen_layer or is_high_change:
                            # stop updating delta_change
                            cmodule.set_adaptive_params(detect_change=False, **kwargs)
                            # stop updating quantization ranges and stats
                            pmodule.apply(torch.ao.quantization.disable_observer)
                            pmodule.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                            # stop updating parmeters
                            for param in pmodule.parameters(recurse=False):
                                param.requires_update = False
                            #
                            self.__quant_params__.forzen_layer_names_list.append(pname)
                        #
                        num_total_layers += 1
                    #
                #
            #
        #
        self.__quant_params__.forzen_layer_names_list = list(set(self.__quant_params__.forzen_layer_names_list))
        print(f"using adaptive quantization - qconfig_mode:{self.__quant_params__.qconfig_mode} "
              f"median_delta_change:{statistics.median(delta_change_list):.4f} max_delta_change:{max(delta_change_list):.4f} "
              f"num_frozen_layers:{len(self.__quant_params__.forzen_layer_names_list)}/{num_total_layers} "
              f"frozen_layers:{self.__quant_params__.forzen_layer_names_list} ")
    #
