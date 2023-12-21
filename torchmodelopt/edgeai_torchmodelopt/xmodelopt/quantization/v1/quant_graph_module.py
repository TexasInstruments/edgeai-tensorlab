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

import warnings
import torch
import copy
from ....xnn import layers
from ....xnn.utils import AttrDict as Dict
from ....xnn import utils
from .hooked_module import *

class QuantGraphModule(HookedModule):
    def __init__(self, module, quantize_in=True, quantize_out=True):
        super().__init__()
        self.module = module
        self.init_qstate()
        self.num_batches_tracked = -1
        self.iter_in_epoch = -1
        self.epoch = -1
        # these are the blocks whose output we quantize for sure.
        # outputs of other clocks such as Conv2d, ConvTranspose2d, BatchNorm2d, Lindear are quantized conditionally
        self.quantize_out_blocks = (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.Hardtanh, layers.QAct, layers.PAct2,
                                    layers.AddBlock, layers.CatBlock, layers.MultBlock, torch.nn.MaxPool2d, torch.nn.AvgPool2d)

        # this block is not quantized. Also if the next block is this, current block is not quantized
        self.ignore_out_blocks = (layers.NoQAct,torch.nn.Dropout2d)

        # quantize the input to a block (under  a certain conditions of the input was not already quantized)
        self.quantize_in = quantize_in

        # whether to quantize the output prediction module or not
        self.quantize_out = quantize_out

        # TBD: is this required
        # # if the original module has load_weights, add it to the quant module also
        # if hasattr(module, 'load_weights'):
        #     def load_weights(m, state_dict, change_names_dict=None):
        #         xnn.utils.load_weights(m.module, state_dict, change_names_dict=change_names_dict)
        #     #
        #     self.load_weights = types.MethodType(load_weights, self)
        # #


    def init_qstate(self):
        if not hasattr(self, '__qstate__'):
            self.__qstate__ = Dict()
        #
        if 'qparams' not in self.get_qstate():
            self.get_qstate().qparams = Dict()
        #
        if 'qparams_prev' not in self.get_qstate():
            self.get_qstate().qparams_prev = Dict()
        #
        if 'analyzed_graph' not in self.get_qstate():
            self.get_qstate().analyzed_graph = False
        #


    def clear_qstate(self):
        self.__qstate__ = Dict()
        self.init_qstate()


    def get_qstate(self):
        return self.__qstate__


    def forward(self, inputs, *args, **kwargs):
        assert False, 'forward is not defined'


    def update_counters(self, force_update=False):
        self.iter_in_epoch += 1
        if self.training or force_update:
            self.num_batches_tracked += 1
            if self.iter_in_epoch == 0:
                self.epoch += 1.0
            #
        #
    #

    # force_update is used to increment inte counters even in non training
    # used for validation in QuantTestModule
    def analyze_graph(self, inputs, *args, force_update=False, merge_weights=False, clear_qstate=False, **kwargs):
        with torch.no_grad():
            self.init_qstate()
            self.update_counters(force_update=force_update)
            if (self.get_qstate().analyzed_graph == False):
                # forward and analyze
                self.forward_analyze_modules(inputs, *args, **kwargs)
                # analyze the connections
                self.analyze_connections()
                self.get_qstate().analyzed_graph = True

                # merge weights so that weight quantization can be done
                if merge_weights:
                    self.merge_weights()
                #

                if clear_qstate:
                    self.clear_qstate()
                #
            #
        #


    def model_surgery_quantize(self, dummy_input, *args, **kwargs):
        # lear the sates - just to be sure
        self.clear_qstate()
        # analyze
        self.analyze_graph(dummy_input, *args, **kwargs)
        # insert QAct wherever range clipping needs to be done
        self.model_surgery_activations()
        # since we might have added new activations, clear the sates as they may not be valid
        self.clear_qstate()
        # need to call analyze_graph in the derived class
    #

    def model_surgery_activations(self):
        for module_hash, qparams in self.get_qstate().qparams.items():
            module = self.get_module(module_hash)
            if isinstance(module, layers.PAct2):
                pass
            elif qparams.quantize_out:
                if utils.is_activation(module):
                    if isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6)):
                        activation_q = layers.PAct2(signed=False)
                    elif isinstance(module, torch.nn.Hardtanh):
                        activation_q = layers.PAct2(clip_range=(module.min_val, module.max_val))
                    elif isinstance(module, layers.QAct):
                        activation_q = layers.PAct2(signed=None)
                    else:
                        activation_q = layers.PAct2(signed=None)
                    #
                    # replace the existing activation by PAct2
                    parent = utils.get_parent_module(self, module)
                    name = utils.get_module_name(parent, module)
                    activation_q.train(self.training)
                    setattr(parent, name, activation_q)
                elif not hasattr(module, 'activation_q'):
                    activation_q = layers.PAct2(signed=None)
                    activation_q.train(self.training)
                    module.activation_q = activation_q
                #
            elif qparams.quantize_in:
                if not hasattr(module, 'activation_in'):
                    activation_in = layers.PAct2(signed=None, range_shrink_activations=self.range_shrink_activations)
                    activation_in.train(self.training)
                    module.activation_in = activation_in
                #
            else:
                pass
            #
        #
    #


    def train(self, mode=True):
        self.iter_in_epoch = -1
        super().train(mode)


    ################################################################
    def forward_analyze_modules(self, inputs, *args, **kwargs):
        '''
        analyze modules needs a call hook - the call hook does not work with DataParallel.
        So, do the analysis on a copy.
        '''
        self_copy = copy.deepcopy(self)
        self_copy._forward_analyze_modules_impl(inputs, *args, **kwargs)
        self.get_qstate().qparams = self_copy.get_qstate().qparams

    def _forward_analyze_modules_impl(self, inputs, *args, **kwargs):
        self.layer_index = -1
        self.start_call()
        self.add_call_hook(self, self._analyze_modules_op)
        forward_analyze_method_name = kwargs.pop('forward_analyze_method', None)
        if forward_analyze_method_name is not None and hasattr(self.module, forward_analyze_method_name):
            # get the bound method to be used as forward
            forward_analyze_method = getattr(self.module, forward_analyze_method_name)
            output = forward_analyze_method(inputs, *args, **kwargs)
        else:
            output = self.module(inputs, *args, **kwargs)
        #
        self.remove_call_hook(self.module)
        self.finish_call()
        return output

    def _analyze_modules_op(self, op, inputs, *args, **kwargs):
        self.layer_index = self.layer_index + 1
        inputs = utils.squeeze_list2(inputs)
        self.start_node(op)
        self.add_node(op, inputs)
        outputs = op.__forward_orig__(inputs, *args, **kwargs)
        self.add_node(op, inputs, outputs)
        self.finish_node(op, inputs, outputs)
        return outputs

    def add_node(self, module, inputs, outputs=None):
        inputs = self.format_tensors(inputs)
        module_hash = self.module_hash(module)

        if module_hash not in list(self.get_qstate().qparams.keys()):
            self.get_qstate().qparams[module_hash] = Dict()
            self.get_qstate().qparams[module_hash].qrange_w = None
            self.get_qstate().qparams[module_hash].qrange_b = None
            self.get_qstate().qparams[module_hash].qrange_in = []
            self.get_qstate().qparams[module_hash].qrange_out = []
            self.get_qstate().qparams[module_hash].is_input = (self.module is module)
            self.get_qstate().qparams[module_hash].previous_node = []
            self.get_qstate().qparams[module_hash].next_node = []
            self.get_qstate().qparams[module_hash].current_node = module_hash
            self.get_qstate().qparams[module_hash].layer_index = self.layer_index

        current_node = self.get_qstate().qparams[module_hash].current_node
        for inp in inputs:
            if hasattr(inp, 'qparams') and hasattr(inp.qparams, 'last_node'):
                prev_module_hash = inp.qparams.last_node
                prev_module = self.get_module(prev_module_hash)
                previous_node = self.get_qstate().qparams[module_hash].previous_node
                next_node = self.get_qstate().qparams[prev_module_hash].next_node

                if str(inp.qparams.last_node) not in [str(p) for p in previous_node]:
                    self.get_qstate().qparams[module_hash].previous_node += [inp.qparams.last_node]
                if str(current_node) not in [str(n) for n in next_node]:
                    self.get_qstate().qparams[prev_module_hash].next_node += [current_node]

        if outputs is not None:
            outputs = self.format_tensors(outputs)
            for opt in outputs:
                if not hasattr(opt, 'qparams'):
                    opt.qparams = Dict()
                #
                # update last_node if this is not a container module
                # if this is a container module, the last_node would have been already filled in in the last leaf module
                if len(module._modules) == 0:
                    opt.qparams.last_node = current_node
                #


    ################################################################
    def analyze_connections(self):
        first_module = None
        for module_hash, qparams in self.get_qstate().qparams.items():
            module = self.get_module(module_hash)
            if utils.is_conv_deconv_linear(module) or utils.is_normalization(module) or utils.is_activation(module):
                first_module = module if first_module is None else first_module
            #
        #

        for module_hash, qparams in self.get_qstate().qparams.items():
            module = self.get_module(module_hash)
            is_first_module = (first_module is module)
            self._analyse_connections_op(module_hash, module, qparams, is_first_module)
        #

        last_quantize_layer_index = -1
        for module_hash, qparams in self.get_qstate().qparams.items():
            if self.get_qstate().qparams[module_hash].layer_index > last_quantize_layer_index and \
                    self.get_qstate().qparams[module_hash].quantize_out:
                last_quantize_layer_index = self.get_qstate().qparams[module_hash].layer_index
            #
        #
        for module_hash, qparams in self.get_qstate().qparams.items():
            #module = self.get_module(module_hash)
            if self.get_qstate().qparams[module_hash].layer_index == last_quantize_layer_index and \
                    (not self.quantize_out):
                self.get_qstate().qparams[module_hash].quantize_out = False
            #
        #

    def _analyse_connections_op(self, module_hash, module, qparams, is_first_module):
        previous_modules = [self.get_module(p) for p in qparams.previous_node] if len(qparams.previous_node)>0 else []
        next_modules = [self.get_module(n) for n in qparams.next_node] if len(qparams.next_node)>0 else []

        quantize_out = False
        if isinstance(module, self.ignore_out_blocks):
            quantize_out = False
        elif utils.is_activation(module):
            if len(next_modules)==1 and utils.is_activation(next_modules[0]):
                quantize_out = False
            else:
                quantize_out = True
            #
        elif isinstance(module, self.quantize_out_blocks):
            if len(next_modules)==1 and utils.is_activation(next_modules[0]):
                quantize_out = False
            else:
                quantize_out = True
            #
        elif utils.is_normalization(module):
            if len(next_modules)==1 and utils.is_activation(next_modules[0]):
                quantize_out = False
            else:
                quantize_out = True
            #
        elif utils.is_conv(module) or utils.is_deconv(module):
            if len(next_modules)==1 and (utils.is_normalization(next_modules[0]) or utils.is_activation(next_modules[0])):
                quantize_out = False
            else:
                quantize_out = True
            #
        elif utils.is_linear(module):
            if len(next_modules)==1 and (utils.is_normalization(next_modules[0]) or utils.is_activation(next_modules[0])):
                quantize_out = False
            else:
                quantize_out = True
            #
        # elif isinstance(module, (torch.nn.AdaptiveAvgPool2d, torch.nn.Upsample, layers.ResizeTo, torch.nn.Flatten)):
        #     quantize_out = True
        # #

        if len(qparams.previous_node) > 0:
            previous_module_hash = qparams.previous_node[-1]
            previous_module = self.get_module(previous_module_hash)
            previous_module_qparams = self.get_qstate().qparams[previous_module_hash]
            is_input_ignored = isinstance(previous_module, self.ignore_out_blocks)
            is_input_quantized = previous_module_qparams.quantize_out if \
                hasattr(previous_module_qparams, 'quantize_out') else False
        else:
            is_input_ignored = False
            is_input_quantized = False
        #

        quantize_in = self.quantize_in and utils.is_conv_deconv_linear(module) and (not is_input_quantized) and \
                      (not is_input_ignored) and is_first_module
        qparams.quantize_w = utils.is_conv_deconv_linear(module)                                # all conv/deconv layers will be quantized
        qparams.quantize_b = utils.is_conv_deconv_linear(module)                                # all conv/deconv layers will be quantized
        qparams.quantize_out = quantize_out                                                     # selectively quantize output
        qparams.quantize_in = quantize_in                                                       # only top modules's input need to be quantized
        multi_input_blocks = (layers.AddBlock, layers.CatBlock, torch.nn.AdaptiveAvgPool2d)
        qparams.align_in = isinstance(module, multi_input_blocks)                               # all tensors to be made same q at the input
        qparams.scale_in = 64.0 if isinstance(module, torch.nn.AdaptiveAvgPool2d) else 1.0      # additional scaleup to simulate fixed point
        qparams.unquantize_out = qparams.is_input                                               # only top modules's output need to be unquantized
        qparams.is_dwconv = utils.is_dwconv(module)
        qparams.next_modules = next_modules
        qparams.is_first_module = is_first_module


    ################################################################
    def merge_weights(self, make_backup=False):
        assert self.get_qstate().analyzed_graph == True, 'graph must be analyzed before merge_weights()'
        with torch.no_grad():
            for module_hash, qparams in self.get_qstate().qparams.items():
                module = self.get_module(module_hash)
                self._merge_weight_op(module_hash, module, qparams, make_backup)
            #
        #
    #
    def _merge_weight_op(self, module_hash, module, qparams, make_backup):
        is_conv = utils.is_conv_deconv(module)

        # note: we consider merging only if there is a single next node
        next_module = qparams.next_modules[0] if len(qparams.next_modules) == 1 else None
        next_bn = isinstance(next_module, torch.nn.BatchNorm2d) if (next_module is not None) else None

        # if the next module is a bn, appy bn merging step
        if is_conv and next_bn:
            conv = module
            bn = next_module

            # weight/bias
            conv_bias = conv.bias.data if (conv.bias is not None) else 0.0
            bn_weight = bn.weight.data if bn.affine else 1.0
            bn_bias = bn.bias.data if bn.affine else 0.0

            # merged weight and offset
            merged_scale = torch.rsqrt(bn.running_var.data + bn.eps) * bn_weight
            if utils.is_conv(conv):
                merged_scale = merged_scale.view(-1, 1, 1, 1)
            elif utils.is_deconv(conv):
                merged_scale = merged_scale.view(1, -1, 1, 1)
            else:
                assert False, 'unable to merge convolution and BN'
            #
            merged_weight = conv.weight.data * merged_scale
            merged_bias = (conv_bias - bn.running_mean.data) * merged_scale.view(-1) + bn_bias

            # bn is set to unity
            bn.running_mean.data.fill_(0.0)
            bn.running_var.data.fill_(1.0 - bn.eps)
            if bn.affine:
                bn.weight.data.fill_(1.0)
                bn.bias.data.fill_(0.0)
            #

            # copy merged weights to conv
            conv.weight.data.copy_(merged_weight)

            # copy merge bias
            if conv.bias is not None:
                conv.bias.data.copy_(merged_bias)
            elif bn.affine:
                bn.bias.data.copy_(merged_bias.data)
            else:
                warnings.warn('problem detected in conv+bn layer pair: either one of conv.bias or bn.affine is required for successfull calibration - preferably bn.affine')
            #
        #
        return


    ################################################################
    def get_qparams(self, module):
        module_hash = self.module_hash(module)
        return self.get_qstate().qparams[module_hash]


    def get_qparams_prev(self, module):
        module_hash = self.module_hash(module)
        return self.get_qstate().qparams_prev[module_hash] if self.get_qstate().qparams_prev else None


    def start_call(self):
        self.call_count = Dict()


    def finish_call(self):
        self.call_count = None


    def start_node(self, module):
        module_name = self.module_name(module)
        if module_name not in list(self.call_count.keys()):
            self.call_count[module_name] = 0
        #
        return


    def finish_node(self, module, inputs, outputs):
        module_name = self.module_name(module)
        self.call_count[module_name] = self.call_count[module_name] + 1
        return


    def module_hash(self, module):
        '''
        A module may be called multiple times in a model. This module has creates a unique name/hash for each call
        using teh call_count. call_count needs tobe correct for this to work as expected.
        call_count is kep up to date by using start_node() / finish_node() calls.
        '''
        module_name = self.module_name(module)
        module_hash = module_name + '-call:{}'.format(self.call_count[module_name])
        return module_hash


    def module_name(self, module):
        name = None
        for n, m in self.named_modules():
            if m is module:
                name = n
            #
        #
        return name


    def get_module(self, module_hash):
        module_name = module_hash.split('-call:')[0]
        for mname, mod in self.named_modules():
            if module_name == mname:
                return mod
        #
        return None


    def is_last_conv(self, module):
        # implementation is not correct. disable it for the time being
        return False #(module is self.last_conv_linear_module)


    def format_tensors(self, inputs):
        # make a list/tuple if inputs is not. if it is a double list, remove the extra one
        inputs = utils.squeeze_list2(utils.make_list(inputs))
        # remove lists/tuple
        inputs = [ipt for ipt in inputs if utils.is_tensor(ipt)]
        return inputs


    def copy_qparams(self, qparams, inputs):
        qparams_copy = Dict()
        for module_hash, qparam_entry in qparams.items():
            qparams_copy[module_hash] = Dict()
            for key, value in qparam_entry.items():
                # deep copy may not work in some cases, so do it conditionally
                try:
                    qparams_copy[module_hash][key] = copy.deepcopy(value)
                except Exception:
                    qparams_copy[module_hash][key] = value
                #
            #
        #
        return qparams_copy

