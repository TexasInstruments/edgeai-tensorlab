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
from edgeai_torchmodelopt import xnn
from .pixel2pixelnet_utils import *


###########################################
class Pixel2PixelSimpleDecoder(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.pred = xnn.layers.ConvDWSepNormAct2d(input_channels, output_channels, kernel_size=3, \
                                                   normalization=(True, False), activation=(False,False))

    def forward(self, x, x_features, x_list):
        return self.pred(x_features)



###########################################
class Pixel2PixelNet(torch.nn.Module):
    def __init__(self, base_model, DecoderClass, model_config):
        super().__init__()
        self.normalisers = torch.nn.ModuleList([xnn.layers.DefaultNorm2d(i_chan) \
                            for i_chan in model_config.input_channels]) if model_config.normalize_input else None
        self.encoder = base_model
        self.output_channels = model_config.output_channels
        self.num_decoders = len(model_config.output_channels) if (model_config.num_decoders is None) else model_config.num_decoders
        self.split_outputs = model_config.split_outputs
        self.multi_task = xnn.layers.MultiTask(num_splits=self.num_decoders, multi_task_type=model_config.multi_task_type, output_type=model_config.output_type,
                                               multi_task_factors=model_config.multi_task_factors) if model_config.multi_task else None
        self.enable_fp16 = model_config.enable_fp16 if 'enable_fp16' in model_config else False

        #if model_config.freeze_encoder:
            #xnn.utils.freeze_bn(self.encoder)

        assert (self.num_decoders==0 or (self.num_decoders==len(model_config.output_type))), 'num_decoders specified is not correct'
        self.decoders = torch.nn.ModuleList()

        if self.num_decoders == 0:
            self.decoders['0'] = Pixel2PixelSimpleDecoder(model_config.shortcut_channels[-1], sum(model_config.output_channels))
        elif self.num_decoders > 0:
            assert len(model_config.output_type) == len(model_config.output_channels), 'output_types and output_channels should have the same length'

            for o_idx in range(self.num_decoders) :
                model_config_d = model_config.split(o_idx)
                # disable argmax in case multiple decoder are joint into one.
                if (self.num_decoders == 1) and (model_config.output_type is not None):
                    model_config_d.output_type = ','.join(model_config.output_type)
                #
                decoder = DecoderClass(model_config_d)
                #if model_config_d.freeze_decoder:
                    #xnn.utils.freeze_bn(decoder)

                self.decoders.append(decoder)
            #
        #

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1.0-(1e-5))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    @xnn.utils.auto_fp16
    def forward(self, x_inp):
        # BN based normalising
        x_list = [norm(x) for (x, norm) in zip(x_inp, self.normalisers)] if self.normalisers else x_inp

        # base encoder module
        x_features, x_list = self.encoder(x_list)

        x_features_split = self.multi_task(x_features) if self.multi_task else [x_features for _ in range(self.num_decoders)]

        # decoder modules
        x_out = []
        for d_idx, d_name in enumerate(self.decoders):
            decoder = self.decoders[d_idx]
            x_feat = x_features_split[d_idx]
            d_out = decoder(x_inp, x_feat, x_list)
            x_out.append(d_out)

        x_out = xnn.layers.split_output_channels(x_out[0], self.output_channels) if (self.num_decoders <= 1 and self.split_outputs) else x_out
        return x_out

