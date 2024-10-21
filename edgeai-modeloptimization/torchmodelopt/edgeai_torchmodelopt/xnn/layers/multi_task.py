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
import torch.optim as optim
import numpy as np
import sys
import torch.nn.functional as F

#####################################################
class MultiTask(torch.nn.Module):
    '''
    multi_task_type: "grad_norm" "pseudo_grad_norm" "naive_grad_norm" "dwa" "dwa_gradnorm" "uncertainty"
    '''
    def __init__(self, num_splits = 1, multi_task_type=None, output_type=None, multi_task_factors = None):
        super().__init__()

        ################################
        # check args
        assert multi_task_type in (None, 'learned', 'uncertainty', 'grad_norm', 'pseudo_grad_norm','dwa_grad_norm'), 'invalid value for multi_task_type'

        self.num_splits = num_splits
        self.losses_short = None
        self.losses_long = None

        # self.loss_scales = torch.nn.Parameter(torch.ones(num_splits, device='cuda:0'))
        # self.loss_scales = torch.ones(num_splits, device='cuda:0', dtype=torch.float32) #requires_grad=True
        self.loss_scales = torch.ones(num_splits, device='cuda:0', dtype=torch.float32) if multi_task_factors is None else \
                            torch.tensor(multi_task_factors, device='cuda:0', dtype=torch.float32)
        self.loss_offsets = torch.zeros(num_splits, device='cuda:0', dtype=torch.float32) #None
        self.dy_norms_smooth = None
        self.register_backward_hook(self.backward_hook)
        self.alpha = 0.75  #0.75 #0.5 #0.12  #1.5
        self.lr = 1e-4  #1e-2 #1e-3 #1e-4 #1e-5
        self.momentum = 0.9
        self.beta = 0.999
        self.multi_task_type = ('pseudo_grad_norm' if multi_task_type == 'learned' else multi_task_type)
        self.output_type = output_type
        self.long_smooth = 1e-6
        self.short_smooth = 1e-3
        self.eps = 1e-6
        # self.grad_loss = torch.nn.L1Loss()
        self.temperature = 1.0 #2.0
        self.dy_norms = None
        if self.multi_task_type == 'uncertainty':
            self.sigma_factor = torch.zeros(num_splits, device='cuda:0', dtype=torch.float32)  # requires_grad=True
            self.uncertainty_factors = self.loss_scales
            for task_idx in enumerate(num_splits):
                output_type = self.output_type[task_idx]
                discrete_loss = (output_type in ('segmentation', 'classification'))
                self.sigma_factor[task_idx] = (1 if discrete_loss else 2)
                self.loss_scales[task_idx] = (-0.5)*torch.log(self.uncertainty_factors[task_idx]*self.sigma_factor[task_idx])

        if self.multi_task_type in ["grad_norm", "uncertainty"]:
            if self.multi_task_type == 'grad_norm':
                param_groups = [{'params':self.loss_scales}]
            elif self.multi_task_type == 'uncertainty':
                param_groups = [{'params': self.uncertainty_factors}]
            self.gradnorm_optimizer = 'sgd' #'adam' #'sgd'
            if self.gradnorm_optimizer == 'adam':
                self.optimizer = torch.optim.Adam(param_groups, self.lr, betas=(self.momentum, self.beta))
            elif self.gradnorm_optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(param_groups, self.lr, momentum=self.momentum)

    def forward(self, x):
        return torch.stack([x for split in range(self.num_splits)])

    def backward_hook(self, module, dy_list, dx):
        self.dy_norms = torch.stack([torch.norm(dy, p=2) for dy in dy_list]).to('cuda:0')
        if self.dy_norms_smooth is not None:
            self.dy_norms_smooth = self.dy_norms_smooth*(1-self.short_smooth) + self.dy_norms*self.short_smooth
        else:
            self.dy_norms_smooth = self.dy_norms
        # dy_norms_smooth_mean = self.dy_norms_smooth.mean()
        self.update_loss_scale()
        del dx, module, dy_list

    def set_losses(self, losses):
        if self.losses_short is not None:
            self.losses_short = self.losses_short*(1-self.short_smooth) + torch.stack([loss.detach() for loss in losses])*self.short_smooth
            self.losses_long = self.losses_long*(1-self.long_smooth) + torch.stack([loss.detach() for loss in losses])*self.long_smooth
        else:
            self.losses_short = torch.stack([loss.detach() for loss in losses])
            self.losses_long = torch.stack([loss.detach() for loss in losses])

    def get_loss_scales(self):
        return self.loss_scales, self.loss_offsets

    def update_loss_scale(self, model=None, loss_list=None):
        # wc  = model.module.encoder.features.stream0._modules['17'].conv._modules['6'].weight #final common layer
        # dy_list = [torch.autograd.grad(loss_list[index], wc, retain_graph=True)[0] #, create_graph=True
        #                 for index in range(len(loss_list))]
        # dy_norms = torch.stack([torch.norm(dy, p=2) for dy in dy_list])
        #
        # if self.dy_norms_smooth is not None:
        #     self.dy_norms_smooth = self.dy_norms_smooth*(1-self.short_smooth) + dy_norms*self.short_smooth
        # else:
        #     self.dy_norms_smooth = dy_norms
        #
        dy_norms_mean = self.dy_norms.mean().detach()

        dy_norms_smooth_mean = self.dy_norms_smooth.mean()
        inverse_training_rate = self.losses_short / (self.losses_long + self.eps)

        if self.multi_task_type == "grad_norm" :#2nd order update rakes load of time to update
            rel_inverse_training_rate = inverse_training_rate/ inverse_training_rate.mean()
            target_dy_norm = dy_norms_mean * rel_inverse_training_rate**self.alpha
            self.optimizer.zero_grad()
            dy_norms_loss = self.grad_loss(self.dy_norms, target_dy_norm)
            # dy_norms_loss.backward()
            self.loss_scales.grad = torch.autograd.grad(dy_norms_loss, self.loss_scales)[0]
            self.optimizer.step()
            #initializing the optimizer and the loss_scales once again
            self.loss_scales = torch.nn.Parameter(3.0 * self.loss_scales / self.loss_scales.sum())
            param_groups = [{'params':self.loss_scales}]
            # self.optimizer = torch.optim.Adam(param_groups, self.lr, betas=(self.momentum, self.beta))
            self.optimizer = torch.optim.SGD(param_groups, self.lr, momentum=self.momentum)
            del inverse_training_rate, rel_inverse_training_rate, target_dy_norm, dy_norms_loss
            torch.cuda.empty_cache()
            return self.loss_scales

        elif self.multi_task_type == "naive_grad_norm": # special case of pseudo_grad_norm with aplha=1.0
            update_factor =  ((dy_norms_smooth_mean / (self.dy_norms_smooth + self.eps)) * (self.losses_short / (self.losses_long+self.eps)))
            self.loss_scales = self.loss_scales + self.lr*(self.loss_scales* (update_factor-1))
            self.loss_scales = 3.0*self.loss_scales/self.loss_scales.sum()
            del update_factor

        elif self.multi_task_type == "pseudo_grad_norm": #works reasonably well
            rel_inverse_training_rate = inverse_training_rate/ inverse_training_rate.sum()
            target_dy_norm = dy_norms_smooth_mean * rel_inverse_training_rate**self.alpha
            update_factor = (target_dy_norm/(self.dy_norms_smooth + self.eps))
            self.loss_scales = self.loss_scales + self.lr*(self.loss_scales* (update_factor-1))
            self.loss_scales = 3.0*self.loss_scales/self.loss_scales.sum()
            del inverse_training_rate, rel_inverse_training_rate, target_dy_norm, update_factor

        elif self.multi_task_type == "dwa": #update using dynamic weight averaging, doesn't work well because of the drastic updates of weights
            self.loss_scales = 3.0 * F.softmax(inverse_training_rate/self.temperature, dim=0)

        elif self.multi_task_type == "dwa_gradnorm": #update using dynamic weight averaging along with gradient information, have the best results until now
            inverse_training_rate = F.softmax(inverse_training_rate/self.temperature, dim=0)
            target_dy_norm = dy_norms_smooth_mean * inverse_training_rate**self.alpha         #Try to tune the smoothness parameter of the gradient
            update_factor = (target_dy_norm/(self.dy_norms_smooth + self.eps))
            self.loss_scales = self.loss_scales + self.lr*(self.loss_scales* (update_factor-1))
            self.loss_scales = 3.0*self.loss_scales/self.loss_scales.sum()
            # print(self.loss_scales)
        elif self.multi_task_type == "uncertainty":  # Uncertainty based weight update
            self.optimizer.step()
            for task_idx in enumerate(self.num_splits):
                clip_scale = True
                loss_scale = torch.exp(self.uncertainty_factors[task_idx]*(-2))/self.sigma_factor
                self.loss_scales[task_idx] = torch.nn.functional.tanh(loss_scale) if clip_scale else loss_scale
            #
            self.loss_offset = self.uncertainty_factors
        elif self.multi_task_type == "dtp":  #dynamic task priority
            pass
        #
        del dy_norms_smooth_mean, dy_norms_mean


    def find_last_common_weight(self):
        """
        :return: Given a model, we must return the last common layer from the encoder. This is not required for the current implementation. However, may be needed in future.
        """
        pass

#####################################################
def set_losses(module, losses):
    def set_losses_func(op):
        if isinstance(op, MultiTask):
            op.set_losses(losses)
    #--
    module.apply(set_losses_func)


loss_scales, loss_offsets = None, None
def get_loss_scales(module):
    def get_loss_scales_func(op):
        global loss_scales, loss_offsets
        if isinstance(op, MultiTask):
            loss_scales, loss_offsets = op.get_loss_scales()
    #--
    module.apply(get_loss_scales_func)
    return loss_scales, loss_offsets
