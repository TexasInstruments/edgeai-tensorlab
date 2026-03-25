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

###########################################################
# Approximate quantized floating point simulation with gradients.
# Can be used for quantized training of models.
###########################################################

import torch
import numpy as np
import copy
import warnings

from ....xnn import layers
from ....xnn import utils
from .quant_base_module import *

warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)


###########################################################
class QuantTrainModule(QuantBaseModule):
    def __init__(self, module, dummy_input, *args, bitwidth_weights=8, bitwidth_activations=8, per_channel_q=False,
                 histogram_range=True, bias_calibration=False, constrain_weights=None,
                 range_shrink_weights=None, range_shrink_activations=None,
                 power2_weight_range=None, power2_activation_range=None, constrain_bias=None, 
                 quantize_in=True, quantize_out=True, verbose_mode=False, total_epochs=0, **kwargs):
        constrain_weights = (not per_channel_q) if constrain_weights is None else constrain_weights
        super().__init__(module, dummy_input, *args, bitwidth_weights=bitwidth_weights, bitwidth_activations=bitwidth_activations,
                         per_channel_q=per_channel_q, histogram_range=histogram_range, bias_calibration=bias_calibration,
                         constrain_weights=constrain_weights, constrain_bias=constrain_bias,
                         range_shrink_weights=range_shrink_weights, range_shrink_activations=range_shrink_activations,
                         power2_weight_range=power2_weight_range, power2_activation_range=power2_activation_range,
                         model_surgery_quantize=True, quantize_in=quantize_in, quantize_out=quantize_out, verbose_mode=verbose_mode,  **kwargs)
        if not total_epochs:
            raise RuntimeError("total_epochs must be provided")
        #
        self.total_epochs = total_epochs
        self.num_epochs_tracked = 0

    def forward(self, inputs, *args, **kwargs):
        # counters such as num_batches_tracked are used. update them.
        self.update_counters()
        # bitwidth_warmup
        self.adjust_bitwidth()

        # outputs
        outputs = self.module(inputs, *args, **kwargs)
        return outputs

    def train(self, mode: bool = True):
        # put the model in expected mode
        super().train(mode=mode)
        # also freeze the params if required
        if mode is True:
            # set the default epoch at which freeze occurs during training (if missing)
            num_batch_norm_update_epochs = ((self.total_epochs//2)-1)
            num_observer_update_epochs = ((self.total_epochs//2)+1)
            freeze_bn = (self.num_epochs_tracked >= num_batch_norm_update_epochs)
            freeze_observers = (self.num_epochs_tracked >= num_observer_update_epochs)
            if freeze_bn:
                utils.print_once('Freezing BN for subsequent epochs')
            #
            if freeze_observers:
                utils.print_once('Freezing ranges for subsequent epochs')
            #
            self.freeze(freeze_bn=freeze_bn, freeze_observers=freeze_observers)
            self.num_epochs_tracked += 1
        else:
            self.freeze()
        #
        return self
    
    def freeze(self, freeze_bn=True, freeze_observers=True):
        # freeze bn and range after some epochs during quantization
        if freeze_bn:
            utils.freeze_bn(self)
        #
        if freeze_observers:
            layers.freeze_quant_range(self)
        #

    def unfreeze(self, freeze_bn=False, freeze_observers=False):
        self.freeze(freeze_bn=freeze_bn, freeze_observers=freeze_observers)
    
    def adjust_bitwidth(self):
        # adjust bitwidth here, if needed, to do gradual bitwidth adjustment
        pass

    def model_surgery_quantize(self, dummy_input, *args, **kwargs):
        super().model_surgery_quantize(dummy_input, *args, **kwargs)

        def replace_func(op):
            for name, m in op._modules.items():
                if utils.is_conv(m):
                    bias = (m.bias is not None)
                    padding_mode = m.padding_mode
                    new_m = QuantTrainConv2d(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size, stride=m.stride,
                                            padding=m.padding, dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=padding_mode)
                elif utils.is_deconv(m):
                    bias = (m.bias is not None)
                    padding_mode = m.padding_mode
                    new_m = QuantTrainConvTranspose2d(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size, stride=m.stride,
                                            padding=m.padding, output_padding=m.output_padding, groups=m.groups, bias=bias, dilation=m.dilation, padding_mode=padding_mode)
                elif utils.is_linear(m):
                    bias = (m.bias is not None)
                    new_m = QuantTrainLinear(in_features=m.in_features, out_features=m.out_features, bias=bias)
                elif utils.is_bn(m):
                    new_m = QuantTrainBatchNorm2d(num_features=m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine,
                                            track_running_stats=m.track_running_stats)
                elif isinstance(m, layers.PAct2):
                    new_m = QuantTrainPAct2(inplace=m.inplace, signed=m.signed,
                                             bitwidth_weights=self.bitwidth_weights, bitwidth_activations=self.bitwidth_activations,
                                            per_channel_q=self.per_channel_q, range_shrink_activations=self.range_shrink_activations,
                                            power2_weight_range=self.power2_weight_range, power2_activation_range=self.power2_activation_range)
                elif isinstance(m, layers.QAct):
                    new_m = QuantTrainPAct2(signed=None, bitwidth_weights=self.bitwidth_weights, bitwidth_activations=self.bitwidth_activations,
                                             per_channel_q=self.per_channel_q, range_shrink_activations=self.range_shrink_activations,
                                            power2_weight_range=self.power2_weight_range, power2_activation_range=self.power2_activation_range)
                elif isinstance(m, (torch.nn.ReLU, torch.nn.ReLU6)):
                    new_m = QuantTrainPAct2(signed=False, bitwidth_weights=self.bitwidth_weights, bitwidth_activations=self.bitwidth_activations,
                                             per_channel_q=self.per_channel_q, range_shrink_activations=self.range_shrink_activations,
                                            power2_weight_range=self.power2_weight_range, power2_activation_range=self.power2_activation_range)
                else:
                    new_m = None
                #
                if new_m is not None:
                    copy_attr_list = ('weight', 'bias', 'eps', 'clips_act', 'clips_w', 'range_shrink_activations')
                    for attr in dir(m):
                        value = getattr(m,attr)
                        if isinstance(value,torch.Tensor) and value is not None:
                            getattr(new_m,attr).data.copy_(value.data)
                        elif isinstance(value,torch.nn.Module):
                            setattr(new_m, attr, getattr(m,attr))
                        elif attr in copy_attr_list:
                            # copy attributes that need to be copied
                            setattr(new_m, attr, getattr(m, attr))
                        #
                    #
                    new_m.train(m.training)
                    setattr(op, name, new_m)
                #
            #
        #
        # apply recursively
        self.apply(replace_func)

        # clear
        self.clear_qstate()
    #


###########################################################
class QuantTrainParams:
    pass


def get_qparams():
    qparams = QuantTrainParams()
    qparams.inputs = []
    qparams.modules = []
    return qparams


def is_merged_layer(x):
    is_merged = (hasattr(x, 'qparams') and isinstance(x.qparams, QuantTrainParams) and len(x.qparams.modules)>0)
    return is_merged


###########################################################
class QuantTrainConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize_enable = True
        self.bitwidth_weights = None
        self.bitwidth_activations = None
        self.per_channel_q = False

    def forward(self, x):
        is_merged = is_merged_layer(x)
        if is_merged:
           warnings.warn('please see if a PAct can be inserted before this module to collect ranges')
        #

        y = super().forward(x)

        if not self.quantize_enable:
            # if quantization is disabled - return
            return y
        #

        qparams = get_qparams()
        qparams.inputs.append(x)
        qparams.modules.append(self)
        if hasattr(x, 'clips_act'):
            qparams.clips_input = x.clips_act
        #
        y.qparams = qparams
        return y
    #


###########################################################
class QuantTrainConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize_enable = True
        self.bitwidth_weights = None
        self.bitwidth_activations = None
        self.per_channel_q = False

    def forward(self, x):
        is_merged = is_merged_layer(x)
        if is_merged:
           warnings.warn('please see if a PAct can be inserted before this module to collect ranges')
        #

        y = super().forward(x)

        if not self.quantize_enable:
            # if quantization is disabled - return
            return y
        #

        qparams = get_qparams()
        qparams.inputs.append(x)
        qparams.modules.append(self)
        if hasattr(x, 'clips_act'):
            qparams.clips_input = x.clips_act
        #
        y.qparams = qparams
        return y
    #


###########################################################
class QuantTrainLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize_enable = True
        self.bitwidth_weights = None
        self.bitwidth_activations = None
        self.per_channel_q = False

    def forward(self, x):
        is_merged = is_merged_layer(x)
        if is_merged:
           warnings.warn('please see if a PAct can be inserted before this module to collect ranges')
        #

        y = super().forward(x)

        if not self.quantize_enable:
            # if quantization is disabled - return
            return y
        #

        qparams = get_qparams()
        qparams.inputs.append(x)
        qparams.modules.append(self)
        if hasattr(x, 'clips_act'):
            qparams.clips_input = x.clips_act
        #
        y.qparams = qparams
        return y
    #


###########################################################
class QuantTrainBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize_enable = True


    def forward(self, x):
        y = super().forward(x)

        if not self.quantize_enable:
            # if quantization is disabled - return
            return y
        #

        if is_merged_layer(x) and utils.is_conv_deconv(x.qparams.modules[-1]):
            qparams = get_qparams()
            qparams.inputs = [x.qparams.inputs[0], x]
            qparams.modules = [x.qparams.modules[0], self]
            if hasattr(x.qparams, 'clips_input'):
                qparams.clips_input = x.qparams.clips_input
            #
            y.qparams = qparams
        #

        return y
    #


###########################################################
# fake quantized PAct2 for training
class QuantTrainPAct2(layers.PAct2):
    def __init__(self, inplace=False, signed=False, clip_range=None, bitwidth_weights=None, bitwidth_activations=None,
                 per_channel_q=False, range_shrink_activations=layers.PAct2.PACT2_RANGE_SHRINK_DEFAULT, power2_weight_range=True,
                 power2_activation_range=True):
        super().__init__(inplace=inplace, signed=signed, clip_range=clip_range,
                         range_shrink_activations=range_shrink_activations,
                         power2_activation_range=power2_activation_range)

        self.bitwidth_weights = bitwidth_weights
        self.bitwidth_activations = bitwidth_activations
        self.per_channel_q = per_channel_q
        self.power2_weight_range = power2_weight_range

        # quantize_backward_type can be 'ste' or 'qte'
        # - quantize_backward_type == 'ste' will cause backward to happen using unquantized weights/bias
        #   (as the contents of yq is propagated inside the tensor y). Uses: PropagateQuantTensorSTE
        # - quantize_backward_type == 'qte'  allow gradient to flow back through conv using the quantized weights/bias
        #   (as yq is directly propagated then). Uses: PropagateQuantTensorQTE
        self.quantize_backward_type = 'ste'

        self.range_shrink_weights = None
        self.update_activation_range = True
        self.quantize_enable = True
        self.quantize_weights = True
        self.quantize_bias = True
        self.quantize_activations = True
        self.constrain_bias = None
        self.constrain_weights = True
        self.bias_calibration = False
        # constraining of weights at this iteration
        self.constrain_weights_iter = 0
        # start bias constraint at this iteration
        self.constrain_bias_start_iter = 85
        # storing of weights at this iteration
        self.store_weights_iter = 0 #85
        self.verbose_mode = False

    def forward(self, x):
        assert (self.bitwidth_weights is not None) and (self.bitwidth_activations is not None), \
                        'bitwidth_weights and bitwidth_activations must not be None'

        # the pact range update happens here - but range clipping depends on quantize_enable
        y = super().forward(x, update_activation_range=self.update_activation_range, enable=self.quantize_enable)

        if not self.quantize_enable:
            return y
        #

        # previous intermediate outputs and other infoirmation are avaliable
        # for example - conv-bn-relu may need to be merged together.
        is_merged = is_merged_layer(x)
        if is_merged:
            qparams = x.qparams
            xorg = qparams.inputs[0]

            conv, bn = None, None
            # merge weight and bias (if possible) across layers
            if len(qparams.modules) == 2 and utils.is_conv_deconv_linear(qparams.modules[-2]) and isinstance(
                    qparams.modules[-1], torch.nn.BatchNorm2d):
                conv = qparams.modules[-2]
                bn = qparams.modules[-1]
            elif len(qparams.modules) == 1 and utils.is_conv_deconv_linear(qparams.modules[-1]):
                conv = qparams.modules[-1]
            elif len(qparams.modules) == 1 and isinstance(qparams.modules[-1], torch.nn.BatchNorm2d):
                assert False, f'quantization: previous layer is a BN without Conv {qparams.modules} - prease inspect the model carefully'
                bn = qparams.modules[-1]
            #
            else:
                assert False, f'QuantTrainPAct2: both conv & bn layes cannot be None in a merged scenario - prease inspect the model carefully'
            #
            conv, weight, bias = self.merge_quantize_weights(qparams, conv, bn)
        else:
            conv, weight, bias = None, None, None
        #

        if is_merged and utils.is_conv(conv):
            xq = torch.nn.functional.conv2d(xorg, weight, bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        elif is_merged and utils.is_deconv(conv):
            xq = torch.nn.functional.conv_transpose2d(xorg, weight, bias, stride=conv.stride, padding=conv.padding, output_padding=conv.output_padding,
                                                      dilation=conv.dilation, groups=conv.groups)
        elif is_merged and utils.is_linear(conv):
            xq = torch.nn.functional.linear(xorg, weight, bias)
        else:
            xq = x
        #

        if (self.quantize_enable and self.quantize_activations):
            use_per_channel_q = (self.per_channel_q == 'all')
            clip_min, clip_max, scale, scale_inv = self.get_clips_scale_act(xq, use_per_channel_q=use_per_channel_q)
            width_min, width_max = self.get_widths_act()
            # no need to call super().forward here as clipping with width_min/windth_max-1 after scaling has the same effect.
            yq = layers.quantize_dequantize_g(xq, scale, width_min, width_max-1, self.power2_activation_range, 1, 'round_up')
            if self.verbose_mode:
                print(f"layer: {self.name} feature_scale = {scale}")
            #
        else:
            yq = super().forward(xq, update_activation_range=False, enable=True)
        #

        if self.quantize_backward_type == 'ste':
            yq = layers.propagate_quant_ste(y, yq)
        elif self.quantize_backward_type == 'qte':
            yq = layers.propagate_quant_qte(y, yq)
        else:
            assert False, 'quantize_backward_type must be one of ste or qte'
        #

        # pass on the clips to be used in the next quantization
        yq.clips_act = self.get_clips_act()
        return yq
    #


    def apply_constrain_weights(self, merged_weight):
        return utils.constrain_weight(merged_weight)


    def is_constrain_bias_value(self):
        return (self.constrain_bias == ConstrainBiasType.CONSTRAIN_BIAS_TYPE_SATURATE)


    def is_reduce_quantize_weight_scale(self):
        return (self.constrain_bias == ConstrainBiasType.CONSTRAIN_BIAS_TYPE_REDUCE_WEIGHT_SCALE)


    def merge_quantize_weights(self, qparams, conv, bn):
        num_batches_tracked = int(self.num_batches_tracked)
        # constrain/clip weights to reduce the dynamic range of weights
        is_constrain_weights_iter = self.training and (num_batches_tracked == self.constrain_weights_iter)
        # store weights once in training after constraining
        is_store_weights_iter = self.training and (num_batches_tracked == self.store_weights_iter)
        # note we do not modify bias here - but rather the weight scale so that the bias doesn't overflow after scaling.
        # weight scale adjustment according to bias constraint needs to happen for train and val
        is_constrain_bias_iter = True #(not self.training) or (num_batches_tracked >= self.constrain_bias_start_iter)
        # store the constrained bias if needed
        is_store_bias_iter = self.training and (num_batches_tracked == self.constrain_bias_start_iter)
        # splitting the quantized bias values if the quantized bias is large
        bias_split_quant = True

        # merge weight and bias (if possible) across layers
        if conv is not None and bn is not None:
            conv_bias = conv.bias if (conv.bias is not None) else torch.tensor(0.0).to(conv.weight.device)
            #
            bn_weight = bn.weight if (bn.weight is not None) else torch.tensor(0.0).to(bn.running_mean.device)
            bn_bias = bn.bias if (bn.bias is not None) else torch.tensor(0.0).to(bn.running_mean.device)
            #
            merged_scale = bn_weight / torch.sqrt(bn.running_var + bn.eps)
            if utils.is_conv(conv):
                merged_scale = merged_scale.view(-1, 1, 1, 1)
            elif utils.is_deconv(conv):
                merged_scale = merged_scale.view(1, -1, 1, 1)
            else:
                assert False, 'unable to merge convolution and BN'
            #
            merged_bias = (conv_bias - bn.running_mean) * merged_scale.view(-1) + bn_bias
            merged_weight = conv.weight * merged_scale
            #
            merged_scale_sign = merged_scale.sign()
            merged_scale_sign = merged_scale_sign + (merged_scale_sign == 0) # make the 0s in sign to 1
            merged_scale_eps = merged_scale.abs().clamp(min=bn.eps) * merged_scale_sign
            merged_scale_inv = 1.0 / merged_scale_eps
            #
        elif conv is not None:
            merged_weight = conv.weight
            merged_bias = conv.bias if (conv.bias is not None) else torch.zeros(conv.out_channels).to(conv.weight.device)
            merged_scale = 1.0
            merged_scale_inv = 1.0
        elif bn is not None:
            merged_weight = bn.weight if (bn.weight is not None) else torch.zeros(bn.num_features).to(bn.running_mean.device)
            merged_bias = bn.bias if (bn.bias is not None) else torch.zeros(bn.num_features).to(bn.running_mean.device)
        else:
            assert False, f'merge_quantize_weights(): both conv & bn layes cannot be None in a merged scenario - prease inspect the model carefully'
            merged_weight = 0.0
            merged_bias = 0.0
        #
        # quantize weight and bias
        if (conv is not None):
            is_dw = utils.is_dwconv(conv)
            is_deconv = utils.is_deconv(conv)
            use_per_channel_q = (is_dw and self.per_channel_q is True) or (self.per_channel_q == 'all')

            # quantize the weights
            if (self.quantize_enable and self.quantize_weights):
                # clip/constrain the weights
                if self.constrain_weights and is_constrain_weights_iter:
                    with torch.no_grad():
                        # clamp merged weights, invert the bn and copy to conv weight
                        constrained_weight = self.apply_constrain_weights(merged_weight.data)
                        merged_weight.data.copy_(constrained_weight.data)
                    #
                #
                # get the weight scale values (multiple values in case of use_per_channel_q)
                weight_clip_min, weight_clip_max, weight_scale2, weight_scale_inv2 = self.get_clips_scale_w(merged_weight,
                                        use_per_channel_q=use_per_channel_q, is_deconv=is_deconv)

                # in some cases, bias quantization can have additional restrictions if for example,
                # bias that is being added to accumulator is limited to 16bit.
                if self.quantize_enable and self.constrain_bias and is_constrain_bias_iter and self.is_reduce_quantize_weight_scale():
                    # use the bias to determine the bias scale allowed due to additional joint constrains
                    clips_scale_joint = self.get_clips_scale_joint(merged_bias)
                    # get the input scale if it is available
                    clips_scale_input = self.get_clips_scale_input(qparams)
                    if clips_scale_input is not None:
                        # scale factor to be used for bias is the product of scale factors of weight and input
                        # using input_scale, work backwards and find the maximum allowed weight scale
                        scale2_joint = clips_scale_joint[2]
                        scale2_input = clips_scale_input[2]
                        scale2_input = torch.clamp(scale2_input, min=self.eps)
                        scale2_weight_max_joint = scale2_joint / scale2_input
                        # limit the weight scale to maximum allowed weight scale
                        weight_scale2 = torch.min(weight_scale2, scale2_weight_max_joint)
                        weight_scale_inv2 = weight_scale2.pow(-1)
                    #
                #

                # do fake quantization of weights
                width_min, width_max = self.get_widths_w()
                per_channel_q_axis = 1 if is_deconv else 0
                merged_weight = layers.quantize_dequantize_g(merged_weight, weight_scale2, width_min, width_max-1,
                                                             self.power2_weight_range, per_channel_q_axis, 'round_sym')
                if self.verbose_mode:
                    print(f"layer: {self.name} weight_scale = {weight_scale2}")
                #
            #

            # quantize the bias
            if (self.quantize_enable and self.quantize_bias):
                bias_width_min, bias_width_max = self.get_widths_bias()
                bias_clip_min, bias_clip_max, bias_scale2, bias_scale_inv2 = self.get_clips_scale_bias(merged_bias)
                power2_bias_range = (self.power2_weight_range and self.power2_activation_range)
                merged_bias = layers.quantize_dequantize_g(merged_bias, bias_scale2, bias_width_min, bias_width_max - 1,
                                                           power2_bias_range, 0, 'round_sym')

                if is_constrain_bias_iter and self.is_constrain_bias_value():
                    clips_scale_input = self.get_clips_scale_input(qparams)
                    if clips_scale_input is not None and weight_scale2 is not None:
                        scale2_input = clips_scale_input[2]
                        scale2_joint = scale2_input * weight_scale2
                        width_joint_min, width_joint_max = self.get_widths_joint()
                        if scale2_joint == 0:
                            merged_bias.fill_(0.0)
                        elif bias_split_quant:
                            bias_scale_ratio_factor = torch.div(scale2_joint, bias_scale2, rounding_mode='trunc')
                            input_signed = clips_scale_input[0] < 0
                            if self.verbose_mode:
                                print(f'layer: {self.name} input_signed = {input_signed}')
                            #
                            quant_bias_max = (bias_width_max-1) if input_signed else (2*bias_width_max-1)
                            if bias_scale_ratio_factor > quant_bias_max:
                                merged_bias_q = torch.clamp(merged_bias*scale2_joint, width_joint_min, width_joint_max - 1)
                                merged_bias_part_q = torch.div(merged_bias_q, quant_bias_max, rounding_mode='trunc')
                                merged_bias_part_q = torch.clamp(merged_bias_part_q, bias_width_min, bias_width_max - 1)
                                merged_bias_q = merged_bias_part_q * quant_bias_max
                                merged_bias = merged_bias_q / scale2_joint
                            else:
                                merged_bias_q = torch.trunc(merged_bias * scale2_joint)
                                merged_bias = merged_bias_q / scale2_joint
                            #
                        else:
                            merged_bias = layers.quantize_dequantize_g(merged_bias, scale2_joint, width_joint_min, width_joint_max - 1,
                                                                       power2_bias_range, 0, round_type=None)
                        #
                    #
                #
                if self.verbose_mode:
                    print(f"layer: {self.name} bias_scale = {bias_scale2}")
                #
            # #

            # invert the bn operation and store weights/bias
            if self.quantize_enable and self.quantize_weights and is_store_weights_iter:
                conv.weight.data.copy_(merged_weight.data * merged_scale_inv)
            #
            # store the constrained bias if needed
            if self.quantize_enable and self.quantize_bias and is_store_bias_iter and \
                    self.constrain_bias == ConstrainBiasType.CONSTRAIN_BIAS_TYPE_SATURATE:
                if conv.bias is not None:
                    if bn is not None:
                        conv_bias = (merged_bias - bn_bias) * merged_scale_inv.view(-1) + bn.running_mean
                        conv.bias.data.copy_(conv_bias.data)
                    else:
                        conv.bias.data.copy_(merged_bias.data)
                    #
                elif bn is not None and bn.bias is not None:
                    bn_bias = merged_bias + bn.running_mean * merged_scale.view(-1)
                    bn.bias.data.copy_(bn_bias.data)
                #
            #
        #
        return conv, merged_weight, merged_bias


    ###########################################################
    def get_widths_w(self):
        # weights
        bw = (self.bitwidth_weights - 1)
        width_max = np.power(2.0, bw)
        width_min = -width_max
        # return
        return (width_min, width_max)


    def get_clips_w(self, tensor):
        # find the clip values
        w_min, w_max = utils.extrema_fast(tensor.data, range_shrink_percentile=self.range_shrink_weights)
        clip_max = torch.max(torch.abs(w_min), torch.abs(w_max))
        clip_max = torch.clamp(clip_max, min=self.eps)
        clip_max2 = layers.functional.ceil2_g(clip_max) if self.power2_weight_range else clip_max
        clip_min2 = -clip_max2
        return (clip_min2, clip_max2)


    def get_clips_scale_w(self, weight, use_per_channel_q=False, is_deconv=False):
        clip_min, clip_max = self.get_clips_w(weight)
        width_min, width_max = self.get_widths_w()
        scale2 = (width_max / clip_max)
        scale2 = torch.clamp(scale2, min=self.eps)
        scale_inv2 = scale2.pow(-1.0)
        if not use_per_channel_q:
            return (clip_min, clip_max, scale2, scale_inv2)
        #
        # the remaining part of the function is only used in the case of use_per_channel_q
        # compute the per-channel weight scale.
        # restrict the weight scale factor of a channel from becoming extremely large.
        scale_factor_ratio_max = 256 #None
        channels = int(weight.size(1)) if is_deconv else int(weight.size(0))
        scale2_array = torch.zeros(1, channels, 1, 1).to(weight.device) if is_deconv else \
            torch.zeros(channels, 1, 1, 1).to(weight.device)
        scale_inv2_array = torch.zeros(1, channels, 1, 1).to(weight.device) if is_deconv else \
            torch.zeros(channels, 1, 1, 1).to(weight.device)
        for chan_id in range(channels):
            weight_channel = weight[:,chan_id,...] if is_deconv else weight[chan_id]
            _, _, scale2_value, scale_inv2_value = self.get_clips_scale_w(weight_channel)
            scale2_value = torch.min(scale2_value, scale2*scale_factor_ratio_max) \
                if (scale_factor_ratio_max is not None) else scale2_value
            scale2_value = torch.clamp(scale2_value, min=self.eps)
            scale_inv2_value = scale2_value.pow(-1.0)
            if is_deconv:
                scale2_array[0, chan_id, 0, 0] = scale2_value
                scale_inv2_array[0, chan_id, 0, 0] = scale_inv2_value
            else:
                scale2_array[chan_id, 0, 0, 0] = scale2_value
                scale_inv2_array[chan_id, 0, 0, 0] = scale_inv2_value
            #
        #
        return (clip_min, clip_max, scale2_array, scale_inv2_array)


    ###########################################################
    def get_widths_act(self):
        if self.signed is None:
            clip_min, clip_max = self.get_clips_act()
            signed = (clip_min < 0.0)
        else:
            signed = self.signed
        #
        bw = (self.bitwidth_activations - 1) if signed else self.bitwidth_activations
        width_max = np.power(2.0, bw)
        width_min = -width_max if signed else 0.0
        return width_min, width_max


    def get_clips_scale_act(self, tensor=None, use_per_channel_q=False):
        clip_min, clip_max = self.get_clips_act()
        width_min, width_max = self.get_widths_act()
        scale2 = width_max / clip_max
        scale2 = torch.clamp(scale2, min=self.eps)
        scale_inv2 = scale2.pow(-1.0)
        if not use_per_channel_q:
            return (clip_min, clip_max, scale2, scale_inv2)
        #
        # the remaining part of the function is only used in the case of use_per_channel_q
        # compute the per-channel weight scale.
        # restrict the weight scale factor of a channel from becoming extremely large.
        scale_factor_ratio_max = 256  # None
        channels = int(tensor.size(1))
        scale2_array = torch.zeros(1, channels, 1, 1).to(tensor.device)
        scale_inv2_array = torch.zeros(1, channels, 1, 1).to(tensor.device)
        for chan_id in range(channels):
            tensor_channel = tensor[:, chan_id, ...]
            _, _, scale2_value, scale_inv2_value = self.get_clips_scale_act(tensor_channel)
            scale2_value = torch.min(scale2_value, scale2 * scale_factor_ratio_max) \
                if (scale_factor_ratio_max is not None) else scale2_value
            scale2_value = torch.clamp(scale2_value, min=self.eps)
            scale_inv2_value = scale2_value.pow(-1.0)
            scale2_array[0, chan_id, 0, 0] = scale2_value
            scale_inv2_array[0, chan_id, 0, 0] = scale_inv2_value
        #
        return (clip_min, clip_max, scale2_array, scale_inv2_array)


    ###########################################################
    # bias uses the same kind of widths
    get_widths_bias = get_widths_w


    # bias uses the same kind of clips
    get_clips_bias = get_clips_w


    def get_clips_scale_bias(self, bias):
        clip_min, clip_max = self.get_clips_bias(bias)
        width_min, width_max = self.get_widths_bias()
        scale2 = (width_max / clip_max)
        scale2 = torch.clamp(scale2, min=self.eps)
        scale_inv2 = scale2.pow(-1.0)
        return (clip_min, clip_max, scale2, scale_inv2)


    ###########################################################
    def get_clips_scale_joint(self, tensor):
        clip_min, clip_max = self.get_clips_bias(tensor)
        width_min, width_max = self.get_widths_joint()
        scale2 = (width_max / clip_max)
        scale2 = torch.clamp(scale2, min=self.eps)
        scale_inv2 = scale2.pow(-1.0)
        return (clip_min, clip_max, scale2, scale_inv2)


    def get_widths_joint(self):
        bw = (4*self.bitwidth_weights-1) if (self.constrain_bias == ConstrainBiasType.CONSTRAIN_BIAS_TYPE_NONE) \
            else (2*self.bitwidth_weights-1)
        width_max = np.power(2.0, bw)
        width_min = -width_max
        return (width_min, width_max)


    def get_clips_input(self, qparams):
        if hasattr(qparams, 'clips_input'):
            return qparams.clips_input
        else:
            return None
        #

    def get_widths_input(self, clip_min, clip_max):
        signed = (clip_min < 0.0)
        bw = (self.bitwidth_activations - 1) if signed else self.bitwidth_activations
        width_max = np.power(2.0, bw)
        width_min = -width_max if signed else 0.0
        return width_min, width_max


    def get_clips_scale_input(self, qparams):
        clips_input = self.get_clips_input(qparams)
        if clips_input is not None:
            clip_min, clip_max = clips_input
            width_min, width_max = self.get_widths_input(clip_min, clip_max)
            scale2 = (width_max / clip_max)
            scale2 = torch.clamp(scale2, min=self.eps)
            scale_inv2 = scale2.pow(-1.0)
            return (clip_min, clip_max, scale2, scale_inv2)
        else:
            return None
        #


