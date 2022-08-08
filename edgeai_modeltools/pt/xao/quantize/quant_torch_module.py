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

###########################################################
# Approximate quantized floating point simulation with gradients.
# Can be used for quantized training of models.
###########################################################
import copy
import torch

from .. import layers
from .. import utils
from .quant_torch_qconfig import *


###########################################################
class QuantTorchModule(torch.nn.Module):
    def __init__(self, module, dummy_input, *args,  backend='fbgemm', symmetric=True, per_channel_q=False, #'depthwise'
                 with_fakequantize=True, is_qat=False, power2_weight_range=True, power2_activation_range=True,
                 histogram=False, constrain_weights=False, freeze_bn=False, clamp_params=False, **kwargs):
        super().__init__()
        self.dummy_input = dummy_input
        self.backend = backend
        self.with_fakequantize = with_fakequantize
        self.is_qat = is_qat
        self.histogram = histogram
        self.symmetric = symmetric
        # check if we have set the depthwise_only mode for per_channel quantization
        self.per_channel_q_depthwise_only = (per_channel_q == 'depthwise')
        # the following does not include per_channel being used only for depthwise - it will be handled elsewhere
        self.per_channel_q = (per_channel_q is True)
        self.power2_weight_range = power2_weight_range
        self.power2_activation_range = power2_activation_range
        self.constrain_weights = (not per_channel_q) if constrain_weights is None else constrain_weights
        self.freeze_bn = freeze_bn
        self.clamp_params = clamp_params
        self.module = module
        self.module.quant_in = torch.quantization.QuantStub()
        self.module.dequant_out = torch.quantization.DeQuantStub()

    def fuse_model(self, inplace=True):
        if self.is_qat or self.with_fakequantize:
            # fuse in train mode for QAT to retain BNs
            self.train()
        else:
            # fuse in eval mode to merge BNs upfront - typically used in PTQ
            self.eval()
        #
        if hasattr(self.module, 'fuse_model'):
            self.module.fuse_model()
        else:
            device = next(self.module.parameters()).device
            dummy_input = self.dummy_input.to(device=device)
            fuse_list = self._get_fuse_list(self.module, dummy_input)
            self.module = torch.quantization.fuse_modules(self.module, fuse_list, inplace=inplace)
        #
        for p in self.modules():
            for n, m in p.named_children():
                if isinstance(m, layers.AddBlock):
                    setattr(p, n, layers.FloatFunctionalBlock('add'))
                elif isinstance(m, layers.MultBlock):
                    setattr(p, n, layers.FloatFunctionalBlock('mul'))
                elif isinstance(m, layers.CatBlock):
                    setattr(p, n, layers.FloatFunctionalBlock('cat'))
                #
            #
        #
        return self.module

    def prepare(self):
        torch.backends.quantized.engine = self.backend
        qconfig_func = get_custom_qconfig_qat if self.is_qat \
            else (get_custom_qconfig_with_fakequantize if self.with_fakequantize else get_custom_qconfig)
        qconfig_args = dict(histogram=self.histogram, symmetric=self.symmetric, per_channel=self.per_channel_q,
                            power2_weight_range=self.power2_weight_range,
                            power2_activation_range=self.power2_activation_range)
        self.module.qconfig = qconfig_func(**qconfig_args)
        if self.with_fakequantize or self.is_qat:
            torch.quantization.prepare_qat(self.module, inplace=True)
        else:
            torch.quantization.prepare(self.module, inplace=True)
        #
        if self.per_channel_q_depthwise_only:
            self._force_per_channel_depthwise_only(qconfig_func, qconfig_args)
        #
        if self.clamp_params and (not self.constrain_weights):
            self.clamp_params_backup()
        #
        if self.constrain_weights:
            self.apply_constrain_weights()
        #
        return

    def _force_per_channel_depthwise_only(self, qconfig_func, qconfig_args):
        for m in self.modules():
            per_channel_modules = (torch.nn.Conv2d,
                torch.nn.intrinsic.ConvBnReLU2d, torch.nn.intrinsic.ConvBn2d, torch.nn.intrinsic.ConvReLU2d,
                torch.nn.intrinsic.qat.ConvBnReLU2d, torch.nn.intrinsic.qat.ConvBn2d, torch.nn.intrinsic.qat.ConvReLU2d)
            if isinstance(m, per_channel_modules) and m.weight.size()[1] == 1 and hasattr(m, 'qconfig'):
                qconfig_args_depthwise = copy.deepcopy(qconfig_args)
                qconfig_args_depthwise.update(dict(per_channel=True))
                m.qconfig = qconfig_func(**qconfig_args_depthwise)
                if hasattr(m, 'weight_fake_quant'):
                    m.weight_fake_quant = m.qconfig.weight()
                #
                if hasattr(m, 'activation_post_process'):
                    m.activation_post_process = m.qconfig.activation()
                #
            #
        #
        return

    def forward(self, inputs, *args, **kwargs):
        # freeze batchnorms in the model. clamp_params also need this freezing
        if self.freeze_bn or self.clamp_params:
            self.freeze_model(freeze_bn_stats=True)
        #
        inputs = self.module.quant_in(inputs)
        outputs = self.module(inputs, *args, **kwargs)
        outputs = self.module.dequant_out(outputs)
        # clamp the weights to a few quantization delta of the original weights for faster convergence
        # but if constrain_weights is used, we cannot clamp as the weights are significantly modified from the original
        if self.clamp_params and (not self.constrain_weights):
            for n, m in self.module.named_modules():
                self.clamp_module_with_delta(n, m)
            #
        #
        return outputs

    def convert(self):
        torch.quantization.convert(self.module, inplace=True)

    def _get_fuse_list(self, module, dummy_input):
        for name, m in module.named_modules():
            m.__track_modules_name__ = name
        #
        def _track_modules1(m, inp, oup):
            prev_module = inp.__track_modules_m__[-1] if hasattr(inp, '__track_modules_m__') else None
            if prev_module is not None:
                if hasattr(prev_module, '__track_modules_next__'):
                    prev_module.__track_modules_next__.append(m)
                else:
                    prev_module.__track_modules_next__ = [m]
                #
                if hasattr(m, '__track_modules_prev__'):
                    m.__track_modules_prev__.append(prev_module)
                else:
                    m.__track_modules_prev__ = [prev_module]
                #
            #
            if hasattr(oup, '__track_modules_m__'):
                oup.__track_modules_m__.append(m)
            else:
                oup.__track_modules_m__ = [m]
            #
        #
        def _track_modules(m, inp, oup):
            inp = inp if isinstance(inp, (list,tuple)) else [inp]
            oup = inp if isinstance(oup, (list,tuple)) else [oup]
            for input in inp:
                for output in oup:
                    _track_modules1(m, input, output)
                #
            #
        #
        for m in module.modules():
            m.__track_modules_m_hook__ = m.register_forward_hook(_track_modules)
        #
        module(dummy_input)
        # analyze
        fuse_list = []
        for m in module.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                m_next = None
                m_next2 = None
                if hasattr(m, '__track_modules_next__') and len(m.__track_modules_next__) == 1:
                    m_next = m.__track_modules_next__[-1]
                    if hasattr(m_next, '__track_modules_next__') and len(m_next.__track_modules_next__) == 1:
                        m_next2 = m_next.__track_modules_next__[-1]
                    #
                #
                if isinstance(m_next, torch.nn.BatchNorm2d) and isinstance(m_next2, (torch.nn.ReLU,torch.nn.ReLU6)):
                    fuse_list.append([m.__track_modules_name__, m_next.__track_modules_name__, m_next2.__track_modules_name__])
                elif isinstance(m_next, torch.nn.BatchNorm2d):
                    fuse_list.append([m.__track_modules_name__, m_next.__track_modules_name__])
                elif isinstance(m_next, (torch.nn.ReLU,torch.nn.ReLU6)):
                    fuse_list.append([m.__track_modules_name__, m_next.__track_modules_name__])
                #
            # elif isinstance(m, layers.FloatFunctionalBlock):
            #     if isinstance(m_next, (torch.nn.ReLU,torch.nn.ReLU6)):
            #         fuse_list.append([m.__track_modules_name__, m_next.__track_modules_name__])
            #     #
            # #
        #
        for m in module.modules():
            if hasattr(m, '__track_modules_m_hook__'):
                m.__track_modules_m_hook__.remove()
                del m.__track_modules_m_hook__
            #
            if hasattr(m, '__track_modules_m__'):
                del m.__track_modules_m__
            #
            if hasattr(m, '__track_modules_prev__'):
                del m.__track_modules_prev__
            #
            if hasattr(m, '__track_modules_next__'):
                del m.__track_modules_next__
            #
            if hasattr(m, '__track_modules_name__'):
                del m.__track_modules_name__
            #
        #
        return fuse_list

    def freeze_model(self, disable_observer=False, freeze_bn_stats=True):
        if disable_observer:
            # Freeze quantizer parameters
            self.module.apply(torch.quantization.disable_observer)
        #
        if freeze_bn_stats:
            # Freeze batch norm mean and variance estimates
            self.module.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        #

    def apply_constrain_weights(self):
        for n, m in self.module.named_modules():
            if isinstance(m, (torch.nn.intrinsic.ConvBn2d, torch.nn.intrinsic.ConvBnReLU2d)):
                running_std = torch.sqrt(m.bn.running_var + m.bn.eps)
                scale_factor = m[1].weight / running_std
                scaled_weight = m[0].weight * scale_factor.reshape([-1, 1, 1, 1])
                clamped_weight = utils.constrain_weight(scaled_weight)
                unscaled_weight = clamped_weight / scale_factor.reshape([-1, 1, 1, 1])
                m.weight.data.copy_(unscaled_weight)
            elif isinstance(m, (torch.nn.intrinsic.qat.ConvBn2d, torch.nn.intrinsic.qat.ConvBnReLU2d)):
                running_std = torch.sqrt(m.bn.running_var + m.bn.eps)
                scale_factor = m.bn.weight / running_std
                scaled_weight = m.weight * scale_factor.reshape([-1, 1, 1, 1])
                clamped_weight = utils.constrain_weight(scaled_weight)
                unscaled_weight = clamped_weight / scale_factor.reshape([-1, 1, 1, 1])
                m.weight.data.copy_(unscaled_weight)
            elif isinstance(m, torch.nn.Conv2d):
                clamped_weight = utils.constrain_weight(m.weight)
                m.weight.data.copy_(clamped_weight)
            #
        #

    def clamp_params_backup(self):
        self.parameters_backup = dict()
        for n, m in self.module.named_modules():
            if isinstance(m, (torch.nn.intrinsic.ConvBn2d, torch.nn.intrinsic.ConvBnReLU2d)):
                running_std = torch.sqrt(m.bn.running_var + m.bn.eps)
                scale_factor = m[1].weight / running_std
                scaled_weight = m[0].weight * scale_factor.reshape([-1, 1, 1, 1])
                self.parameters_backup[n] = scaled_weight
            elif isinstance(m, (torch.nn.intrinsic.qat.ConvBn2d, torch.nn.intrinsic.qat.ConvBnReLU2d)):
                running_std = torch.sqrt(m.bn.running_var + m.bn.eps)
                scale_factor = m.bn.weight / running_std
                scaled_weight = m.weight * scale_factor.reshape([-1, 1, 1, 1])
                self.parameters_backup[n] = scaled_weight
            elif isinstance(m, torch.nn.Conv2d):
                self.parameters_backup[n] = m.weight
            #

    def clamp_module_with_delta(self, n, m):
        if isinstance(m, (torch.nn.intrinsic.ConvBn2d, torch.nn.intrinsic.ConvBnReLU2d)):
            running_std = torch.sqrt(m.bn.running_var + m.bn.eps)
            scale_factor = m[1].weight / running_std
            scaled_weight = m[0].weight * scale_factor.reshape([-1, 1, 1, 1])
            scaled_weight_start = self.parameters_backup[n]
            clamped_weight = self.clamp_param_with_delta(scaled_weight, scaled_weight_start)
            unscaled_weight = clamped_weight / scale_factor.reshape([-1, 1, 1, 1])
            m[0].weight.data.copy_(unscaled_weight)
        elif isinstance(m, (torch.nn.intrinsic.qat.ConvBn2d, torch.nn.intrinsic.qat.ConvBnReLU2d)):
            running_std = torch.sqrt(m.bn.running_var + m.bn.eps)
            scale_factor = m.bn.weight / running_std
            scaled_weight = m.weight * scale_factor.reshape([-1, 1, 1, 1])
            scaled_weight_start = self.parameters_backup[n]
            clamped_weight = self.clamp_param_with_delta(scaled_weight, scaled_weight_start)
            unscaled_weight = clamped_weight / scale_factor.reshape([-1, 1, 1, 1])
            m.weight.data.copy_(unscaled_weight)
        elif isinstance(m, torch.nn.Conv2d):
            weight_start = self.parameters_backup[n]
            clamped_weight = self.clamp_param_with_delta(m.weight, weight_start)
            m.weight.data.copy_(clamped_weight)
        #

    def clamp_param_with_delta(self, p, p_start):
        # clamp the weights within a few quatization delta step of the original weights
        # weight is a signed quantity, so 1.0/128.0 is one quantization delta
        p_max = torch.max(torch.abs(p_start.data))
        p_delta = p_max * 2.0 / 128.0
        p_new = torch.min(torch.max(p.data, p_start.data - p_delta), p_start.data + p_delta)
        return p_new