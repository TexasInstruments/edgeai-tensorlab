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

# set bn's to eval in order to preserve statistics - otherwise it will be difficult to implement.
# not setting the whole model to eval as the behavior of some models may not be same.
# also the pact modules need range update, which won't happen in eval mode.
def freeze_bn(module):
    def _freeze_bn_op(op):
        if isinstance(op, torch.nn.BatchNorm2d):
            # method 1 - set to use batch statistics - not correct
            # op.track_running_stats = False
            #
            # method 2 - set bn to eval
            # it is possible that this bn has activations (inserted for quantization)
            # # in that case, do not change their train/eval state
            training = module.training
            op.eval()
            for sub_op in op.children():
                sub_op.train(training)
            #
        #

    #
    module.apply(_freeze_bn_op)


def unfreeze_bn(module):
    def _unfreeze_bn_op(op):
        if isinstance(op, torch.nn.BatchNorm2d):
            # method 1 - change back from use of batch statistics - not correct
            # op.track_running_stats = True
            #
            # method 2 - set bn to train
            op.train()
        #

    #
    module.apply(_unfreeze_bn_op)

