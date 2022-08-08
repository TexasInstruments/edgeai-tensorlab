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

from .quant_torch_qconfig_qat import *
from .quant_torch_eagertrain_module import *

#warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)


###########################################################
class QuantTorchEagerDistillModule(QuantTorchEagerTrainModule):
    def __init__(self, module, dummy_input, *args, freeze_bn=True, clamp_params=True, learning_rate=1e-4,
                 momentum=0.9, weight_decay=4e-5, loss_type=None, aux_loss_weight=0.1, retain_graph=False, **kwargs):
        super().__init__(module, dummy_input, *args, freeze_bn=freeze_bn, clamp_params=clamp_params, **kwargs)
        self.learning_rate = learning_rate
        self.momentum= momentum
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.SGD(self.module.parameters(), self.learning_rate,
                        momentum=self.momentum, weight_decay=self.weight_decay)
        self.set_learning_rate(learning_rate=learning_rate)
        # make a float copy of the core module before it is quantized
        # float copy is used to make reference outputs for distillation
        device = next(self.module.parameters()).device
        self.module_float = copy.deepcopy(self.module)
        self.module_float.to(device=device)
        self.module_float.eval()
        # distillation function
        self.loss_type = loss_type
        self.aux_loss_weight = aux_loss_weight
        self.aud_loss_input = True
        self.retain_graph = retain_graph

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        #

    def distill_criterion(self, student_tensor, teacher_tensor, T=4.0):
        if self.loss_type == 'kl_divergence':
            student_tensor = torch.log(torch.softmax(student_tensor / T, dim=1))
            teacher_tensor = torch.log(torch.softmax(teacher_tensor / T, dim=1))
            loss = torch.nn.functional.kl_div(student_tensor, teacher_tensor, reduction='batchmean', log_target=False)
            loss = loss * T * T / student_tensor.shape[0]
        elif self.loss_type == 'mse':
            loss = torch.nn.functional.mse_loss(student_tensor, teacher_tensor)
        else:
            loss = torch.nn.functional.smooth_l1_loss(student_tensor, teacher_tensor)
        #
        return loss

    def fuse_model(self, inplace=True):
        self.module_float.load_state_dict(self.module.state_dict())
        super().fuse_model(inplace=inplace)

    def prepare(self):
        super().prepare()
        if self.aux_loss_weight is not None:
            for m in self.module.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.register_forward_hook(self.collect_activation_quant_hook)
                #
            #
            for m in self.module_float.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.register_forward_hook(self.collect_activation_float_hook)
                #
            #
        #

    def collect_activation_quant_hook(self, m, inp, oup):
        act_tensor = inp if self.aud_loss_input else oup
        act_tensor = act_tensor[0] if isinstance(act_tensor, (list,tuple)) and len(act_tensor) == 1 else act_tensor
        self.activations_quant.append(act_tensor)

    def collect_activation_float_hook(self, m, inp, oup):
        act_tensor = inp if self.aud_loss_input else oup
        act_tensor = act_tensor[0] if isinstance(act_tensor, (list,tuple)) and len(act_tensor) == 1 else act_tensor
        self.activations_float.append(act_tensor)

    def forward(self, inputs, *args, **kwargs):
        self.activations_quant = []
        self.activations_float = []

        if self.training:
            # make sure that module_float is in eval mode
            self.module_float.eval()
        #

        # output with fake quantization
        outputs = super().forward(inputs, *args, **kwargs)

        if self.training:
            # do the forward of module_float and create reference output
            with torch.no_grad():
                outputs_float = self.module_float(inputs, *args, **kwargs)
            #
            loss = self.distill_criterion(outputs, outputs_float.detach())
            if self.aux_loss_weight is not None:
                # compute the loss w.r.t. float output and do backpropagation
                for oquant, ofloat in zip(self.activations_quant, self.activations_float):
                    loss = loss + self.distill_criterion(oquant, ofloat.detach()) * self.aux_loss_weight
                #
            #
            self.optimizer.zero_grad()
            loss.backward(retain_graph=self.retain_graph)
            self.optimizer.step()
        #
        return outputs

