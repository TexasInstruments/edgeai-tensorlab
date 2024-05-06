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


import os
import collections
import re
import torch

source_checkpoint = '/data/ssd/files/a0393608/work/code/ti/edgeai-algo/edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/retinanet-lite_regnetx-800mf_fpn_bgr_512x512_20200908_checkpoint.pth'

dest_checkoint = os.path.splitext(os.path.basename(source_checkpoint))
dest_checkoint = dest_checkoint[0] + '_converted' + dest_checkoint[1]

checkpoint_dict = torch.load(source_checkpoint)
state_dict = checkpoint_dict['state_dict']


change_names_dict_ssd = {
    r'lateral_convs.(\d).0.': r'lateral_convs.\1.conv.',
    r'lateral_convs.(\d).1.': r'lateral_convs.\1.bn.',

    r'fpn_convs.(\d).0.0.': r'fpn_convs.\1.conv.0.0.',
    r'fpn_convs.(\d).0.1.': r'fpn_convs.\1.conv.0.1.',
    r'fpn_convs.(\d).1.0.': r'fpn_convs.\1.conv.1.0.',
    r'fpn_convs.(\d).1.1.': r'fpn_convs.\1.bn.',

    r'bbox_head.cls_convs.(\d).(\d).(\d).': r'bbox_head.cls_convs.\1.0.\2.\3.',
    r'bbox_head.reg_convs.(\d).(\d).(\d).': r'bbox_head.reg_convs.\1.0.\2.\3.',
}

# change_names_dict_yolo = {
#     r'neck.conv(\d).0.': r'neck.conv\1.conv.',
#     r'neck.conv(\d).1.': r'neck.conv\1.bn.',
#
#     r'neck.detect(\d).conv(\d).0.0.': r'neck.detect\1.conv\2.conv.0.0.',
#     r'neck.detect(\d).conv(\d).0.1.': r'neck.detect\1.conv\2.conv.0.1.',
#     r'neck.detect(\d).conv(\d).1.0.': r'neck.detect\1.conv\2.conv.1.0.',
#     r'neck.detect(\d).conv(\d).1.1.': r'neck.detect\1.conv\2.bn.',
#
#     r'neck.detect(\d).conv(\d).0.': r'neck.detect\1.conv\2.conv.',
#     r'neck.detect(\d).conv(\d).1.': r'neck.detect\1.conv\2.bn.',
#
#     r'bbox_head.convs_bridge.(\d).0.0.': r'bbox_head.convs_bridge.\1.conv.0.0.',
#     r'bbox_head.convs_bridge.(\d).0.1.': r'bbox_head.convs_bridge.\1.conv.0.1.',
#     r'bbox_head.convs_bridge.(\d).1.0.': r'bbox_head.convs_bridge.\1.conv.1.0.',
#     r'bbox_head.convs_bridge.(\d).1.1.': r'bbox_head.convs_bridge.\1.bn.',
# }


# change_names_dict_retinanet = {
#     r'lateral_convs.(\d).0.': r'lateral_convs.\1.conv.',
#     r'lateral_convs.(\d).1.': r'lateral_convs.\1.bn.',
#
#     r'fpn_convs.(\d).0.0.': r'fpn_convs.\1.conv.0.0.',
#     r'fpn_convs.(\d).0.1.': r'fpn_convs.\1.conv.0.1.',
#     r'fpn_convs.(\d).1.0.': r'fpn_convs.\1.conv.1.0.',
#     r'fpn_convs.(\d).1.1.': r'fpn_convs.\1.bn.',
#
#     r'bbox_head.cls_convs.(\d).0.0.': r'bbox_head.cls_convs.\1.conv.0.0.',
#     r'bbox_head.cls_convs.(\d).0.1.': r'bbox_head.cls_convs.\1.conv.0.1.',
#     r'bbox_head.cls_convs.(\d).1.0.': r'bbox_head.cls_convs.\1.conv.1.0.',
#     r'bbox_head.cls_convs.(\d).1.1.': r'bbox_head.cls_convs.\1.bn.',
#
#     r'bbox_head.reg_convs.(\d).0.0.': r'bbox_head.reg_convs.\1.conv.0.0.',
#     r'bbox_head.reg_convs.(\d).0.1.': r'bbox_head.reg_convs.\1.conv.0.1.',
#     r'bbox_head.reg_convs.(\d).1.0.': r'bbox_head.reg_convs.\1.conv.1.0.',
#     r'bbox_head.reg_convs.(\d).1.1.': r'bbox_head.reg_convs.\1.bn.',
# }

#change_names_dict_retinanet #change_names_dict_ssd #change_names_dict_yolo
change_names_dict = change_names_dict_ssd


state_dict_new = collections.OrderedDict()
for k, v in state_dict.items():
    for sk in change_names_dict:
        if re.search(re.compile(sk), k):
            new_k = re.sub(sk, change_names_dict[sk], k)
            break
        else:
            new_k = k
        #
    #
    print(f'{k} -> {new_k}')
    state_dict_new[new_k] = v
#

checkpoint_dict['state_dict'] = state_dict_new
torch.save(checkpoint_dict, dest_checkoint)

