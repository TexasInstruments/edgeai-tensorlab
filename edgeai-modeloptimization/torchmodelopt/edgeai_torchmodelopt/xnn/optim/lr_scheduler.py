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

import numpy as np
import torch


class MultiStepLRWarmup(torch.optim.lr_scheduler.MultiStepLR):
    def __init__(self, *args, warmup_epochs=5, warmup_ratio=1e-2, **kwargs):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        super().__init__(*args, **kwargs)

    def get_lr(self):
        if self.last_epoch == 0:
            return [lr * self.warmup_ratio for lr in self.base_lrs]
        elif self.last_epoch < self.warmup_epochs:
            return [lr * self.last_epoch / self.warmup_epochs for lr in self.base_lrs]
        elif self.last_epoch == self.warmup_epochs:
            return self._get_closed_form_lr()
        else:
            return super().get_lr()


class CosineAnnealingLRWarmup(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, *args, warmup_epochs=5, warmup_ratio=1e-2, **kwargs):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        super().__init__(*args, **kwargs)

    def get_lr(self):
        if self.last_epoch == 0:
            return [lr * self.warmup_ratio for lr in self.base_lrs]
        elif self.last_epoch < self.warmup_epochs:
            return [lr * self.last_epoch / self.warmup_epochs for lr in self.base_lrs]
        elif self.last_epoch == self.warmup_epochs and hasattr(self, "_get_closed_form_lr"):
            return self._get_closed_form_lr()
        else:
            return super().get_lr()


class SchedulerWrapper(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, scheduler_type, optimizer, epochs, start_epoch=0, warmup_epochs=5, warmup_factor=None,
                 max_iter=None, polystep_power=1.0, milestones=None, multistep_gamma=0.5):

        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor

        if scheduler_type == 'step':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=multistep_gamma, last_epoch=start_epoch-1)
        elif scheduler_type == 'poly':
            lambda_scheduler = lambda last_epoch: ((1.0-last_epoch/epochs)**polystep_power)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_scheduler, last_epoch=start_epoch-1)
        elif scheduler_type == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=start_epoch-1)
        else:
            raise ValueError('Unknown scheduler {}'.format(scheduler_type))
        #
        self.lr_scheduler = lr_scheduler
        self.lr_backup = [param_group['lr'] for param_group in self.lr_scheduler.optimizer.param_groups]
        if start_epoch > 0:
            # adjust the leraning rate to that of the start_epoch
            for step in range(start_epoch):
                self.step()
            #
        else:
            # to take care of first iteration and set warmup lr in param_group
            self.get_lr()
        #


    def get_lr(self):
        epoch = self.lr_scheduler.last_epoch
        if self.warmup_epochs and epoch <= self.warmup_epochs:
            lr = [(epoch * l_rate / self.warmup_epochs) for l_rate in self.lr_scheduler.base_lrs]
            if epoch == 0 and self.warmup_factor is not None:
                warmup_lr = [w_rate*self.warmup_factor for w_rate in self.lr_scheduler.base_lrs]
                lr = [max(l_rate, w_rate) for l_rate, w_rate in zip(lr,warmup_lr)]
            #
        else:
            lr = [param_group['lr'] for param_group in self.lr_scheduler.optimizer.param_groups]
        #
        lr = [max(l_rate,0.0) for l_rate in lr]
        for param_group, l_rate in zip(self.lr_scheduler.optimizer.param_groups, lr):
            param_group['lr'] = l_rate
        #
        return lr


    def step(self):
        # some of the scheduler implementations in torch.option may be recursive (depends on previous lr) eg. cosine
        # so it is necessary to restore the original lr from scheduler
        for param_group, l_rate in zip(self.lr_scheduler.optimizer.param_groups, self.lr_backup):
            param_group['lr'] = l_rate
        #
        # actual scheduler call
        self.lr_scheduler.step()
        # backup the lr from scheduler
        self.lr_backup = [param_group['lr'] for param_group in self.lr_scheduler.optimizer.param_groups]
        # return the lr - warmup will be applied in this step
        return self.get_lr()


    def load_state_dict(self, state):
        self.lr_scheduler.load_state_dict(state)


    def state_dict(self):
        return self.lr_scheduler.state_dict()

