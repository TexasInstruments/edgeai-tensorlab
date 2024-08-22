# Copyright (c) 2018-2021, Texas Instruments
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

import copy
import torch
import mmcv
from mmcv.cnn import bricks
import edgeai_torchmodelopt


__all__ = [
    'convert_to_lite_model'
]


def replace_maxpool2d(m):
    from mmdet.models.backbones.csp_darknet import SequentialMaxPool2d
    if m.kernel_size > 3:
        new_m = SequentialMaxPool2d(m.kernel_size, m.stride)
    else:
        new_m = m
    #
    return new_m


def convert_to_lite_model(model, cfg):
    from mmdet.models.backbones.csp_darknet import Focus, FocusLite
    if hasattr(cfg,'convert_to_lite_model') : 
        convert_to_lite_model_args = copy.deepcopy(cfg.convert_to_lite_model)
        convert_to_lite_model_args.pop('model_surgery', None)
    else:
        convert_to_lite_model_args = dict()
    replacement_dict = copy.deepcopy(edgeai_torchmodelopt.xmodelopt.surgery.v1.get_replacement_dict_default(**convert_to_lite_model_args))
    replacements_ext = {
        'mmdet_convert_focus_to_focuslite': {Focus:[FocusLite, 'in_channels', 'out_channels', 'kernel_size', 'stride']},
        'mmdet_convert_swish_to_relu': {bricks.Swish:[torch.nn.ReLU]},
        'mmdet_break_maxpool2d_with_kernel_size_greater_than_equalto_5': {torch.nn.MaxPool2d:[replace_maxpool2d]}
    }
    replacement_dict.update(replacements_ext)
    model = edgeai_torchmodelopt.xmodelopt.surgery.v1.convert_to_lite_model(model, replacement_dict=replacement_dict,
                                                    **convert_to_lite_model_args)
    return model
