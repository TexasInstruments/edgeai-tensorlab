#################################################################################
# Modified from: https://github.com/pytorch/vision
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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
#################################################################################

import torch
import warnings

from collections import OrderedDict
from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Dict

from . import _utils as det_utils
from .ssd import SSD, SSDScoringHead
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .. import mobilenet, regnet, efficientnet
from ..._internally_replaced_utils import load_state_dict_from_url
from .backbone_utils import BackboneWithFPN
from ...ops.feature_pyramid_network import LastLevelP6P7, FeaturePyramidNetwork
from ...ops.rf_blocks import BiFPN
from ...edgeailite import xnn
from .ssdlite import _normal_init, SSDLiteHead, _load_state_dict, model_urls


__all__ = ['ssdlite_mobilenet_v2_fpn', 'ssdlite_mobilenet_v3_large_fpn', 'ssdlite_mobilenet_v3_small_fpn',
           'ssdlite_regnet_x_400mf_fpn', 'ssdlite_regnet_x_800mf_fpn', 'ssdlite_regnet_x_1_6gf_fpn',
           'ssdlite_efficientnet_b0_fpn', 'ssdlite_efficientnet_b2_fpn',
           'ssdlite_efficientnet_b0_bifpn', 'ssdlite_efficientnet_b2_bifpn']


def ssdlite_fpn_model(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                      pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                      norm_layer: Optional[Callable[..., nn.Module]] = None,
                      size: Optional[Tuple] = None,
                      backbone_name = None,
                      shortcut_layers = ('7', '14', '16'),
                      shortcut_channels = (80, 160, 960),
                      fpn_channels=256,
					  fpn_type=FeaturePyramidNetwork,
                      weights_name = None,
                      **kwargs: Any):
    """Constructs an SSDlite model and a MobileNetV3 Large backbone with FPN

    Example:

        >>> model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 320, 320), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable.
        norm_layer (callable, optional): Module specifying the normalization layer to use.
    """
    if size is None:
        warnings.warn("The size of the model is not provided; using default.")
        size = (320, 320)

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    # reduce_tail = not pretrained_backbone

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03) #nn.BatchNorm2d

    if 'mobilenet' in backbone_name:
        backbone_module = mobilenet
    elif 'regnet' in backbone_name:
        backbone_module = regnet
    elif 'efficientnet' in backbone_name:
        backbone_module = efficientnet
    else:
        assert False, f'unknown backbone model name {backbone_name}'

    backbone = backbone_module.__dict__[backbone_name](pretrained=pretrained_backbone, progress=progress,
                                                       norm_layer=norm_layer)

    if not pretrained_backbone:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)

    return_layers = {s:s for s in shortcut_layers}
    # in FeaturePyramidNetwork extra_blocks are applied at the output.
    # in BiFPN they are applied at the input
    p5_channels = fpn_channels if fpn_type == FeaturePyramidNetwork else shortcut_channels[-1]
    extra_blocks = LastLevelP6P7(p5_channels,fpn_channels,num_blocks=3)

    backbone_features = backbone.features if hasattr(backbone, 'features') else backbone
    backbone = BackboneWithFPN(backbone_features, return_layers, in_channels_list=shortcut_channels,
                               out_channels=fpn_channels, extra_blocks=extra_blocks,
                               fpn_type=fpn_type)

    # find the index of the layer from which we wont freeze
    if trainable_backbone_layers is not None:
        backbone_modules = list(backbone.modules())
        trainable_layers = min(trainable_backbone_layers, len(backbone_modules))
        freeze_before = len(backbone_modules) - trainable_layers
        for b in backbone_modules[:freeze_before]:
            for parameter in b.parameters():
                parameter.requires_grad_(False)

    # these factors are chosen to match those in mmdetection
    aspect_ratios = [[2]] + [[2,3]]*4 + [[2]]
    anchor_generator = DefaultBoxGenerator(aspect_ratios, min_ratio=0.1, max_ratio=0.9)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.02,
        "nms_thresh": 0.45,
        "detections_per_img": 300,
        "topk_candidates": 1000, #300
        # If image_mean or image_std is in kwargs, use it, otherwise these defaults will take effect
        # The following default mean/std rescale the input in a way compatible to the (tensorflow?) backbone:
        # i.e. rescale the data in [0, 1] to [-1, -1]. But this is different from typical torchvision backbones.
        "image_mean": kwargs.pop('image_mean', [0.5, 0.5, 0.5]),
        "image_std":  kwargs.pop('image_std', [0.5, 0.5, 0.5]),
    }
    kwargs = {**defaults, **kwargs}
    ssd_head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)
    model = SSD(backbone, anchor_generator, size, num_classes, head=ssd_head, **kwargs)

    if pretrained is True:
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        model.load_state_dict(state_dict)
    elif xnn.utils.is_url(pretrained):
        state_dict = load_state_dict_from_url(pretrained, progress=progress)
        _load_state_dict(model, state_dict)
    elif isinstance(pretrained, str):
        state_dict = torch.load(pretrained)
        _load_state_dict(model, state_dict)
    return model


def _load_state_dict(model, state_dict):
    state_dict = state_dict['model'] if 'model' in state_dict else state_dict
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict(state_dict, strict=False)


###################################################################################
def ssdlite_mobilenet_v2_fpn(*args, backbone_name="mobilenet_v2", **kwargs):
    return ssdlite_fpn_model(*args, backbone_name=backbone_name, shortcut_layers=('6', '10', '17'),
                             shortcut_channels=(32, 64, 320), **kwargs)


###################################################################################
def ssdlite_mobilenet_v3_large_fpn(*args, backbone_name="mobilenet_v3_large", **kwargs):
    return ssdlite_fpn_model(*args, backbone_name=backbone_name, shortcut_layers=('6', '12', '15'),
                             shortcut_channels=(40, 112, 160), **kwargs)


def ssdlite_mobilenet_v3_small_fpn(*args, backbone_name="mobilenet_v3_small", **kwargs):
    return ssdlite_fpn_model(*args, backbone_name=backbone_name, shortcut_layers=('6', '12', '15'),
                             shortcut_channels=(40, 112, 96), **kwargs)


###################################################################################
def ssdlite_regnet_x_400mf_fpn(*args, backbone_name="regnet_x_400mf", **kwargs):
    # shortcut_channels need not be populated for FPN
    return ssdlite_fpn_model(*args, backbone_name=backbone_name,
                             shortcut_layers=('trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4'),
                             shortcut_channels=(64,160,384), **kwargs)


def ssdlite_regnet_x_800mf_fpn(*args, backbone_name="regnet_x_800mf", **kwargs):
    # shortcut_channels need not be populated for FPN
    return ssdlite_fpn_model(*args, backbone_name=backbone_name,
                             shortcut_layers=('trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4'),
                             shortcut_channels=(128,288,672), **kwargs)


def ssdlite_regnet_x_1_6gf_fpn(*args, backbone_name="regnet_x_1_6gf", **kwargs):
    # shortcut_channels need not be populated for FPN
    return ssdlite_fpn_model(*args, backbone_name=backbone_name,
                             shortcut_layers=('trunk_output.block2', 'trunk_output.block3', 'trunk_output.block4'),
                             shortcut_channels=(168, 408, 912), **kwargs)


###################################################################################
def ssdlite_efficientnet_b0_fpn(*args, backbone_name="efficientnet_b0", **kwargs):
    # shortcut_channels need not be populated for FPN
    return ssdlite_fpn_model(*args, backbone_name=backbone_name,
                             shortcut_layers=('3', '5', '7'),
                             shortcut_channels=(40, 112, 320), **kwargs)


def ssdlite_efficientnet_b2_fpn(*args, backbone_name="efficientnet_b2", **kwargs):
    # shortcut_channels need not be populated for FPN
    return ssdlite_fpn_model(*args, backbone_name=backbone_name,
                             shortcut_layers=('3', '5', '7'),
                             shortcut_channels=(48, 120, 352), **kwargs)


###################################################################################
def ssdlite_efficientnet_b0_bifpn(*args, backbone_name="efficientnet_b0", **kwargs):
    '''
    An SSD model that tries to imitate EfficientDet (https://arxiv.org/abs/1911.09070).
    Key differences:
    - SSD is used here instead of a RetinaNet like meta architecture in EfficientDet.
    - Wider BiFPN (more channels, compared to what is used in the BiFPN of EfficientDet-D0) for faster training convergence.
    - BiFPN used here does not have weighted addition, but only direct addition to be more embedded friendly.
    '''
    BiFPN3 = partial(BiFPN, num_blocks=3)
    return ssdlite_fpn_model(*args, backbone_name=backbone_name,
                             shortcut_layers=('3', '5', '7'),
                             shortcut_channels=(40, 112, 320),
                             fpn_type=BiFPN3, fpn_channels=128, **kwargs)

def ssdlite_efficientnet_b2_bifpn(*args, backbone_name="efficientnet_b2", **kwargs):
    '''
    An SSD model that tries to imitate EfficientDet (https://arxiv.org/abs/1911.09070).
    Key differences:
    - SSD is used here instead of a RetinaNet like meta architecture in EfficientDet.
    - Wider BiFPN (more channels, compared to what is used in the BiFPN of EfficientDet-D2) for faster training convergence.
    - BiFPN used here does not have weighted addition, but only direct addition to be more embedded friendly.
    '''
    BiFPN5 = partial(BiFPN, num_blocks=5)
    return ssdlite_fpn_model(*args, backbone_name=backbone_name,
                             shortcut_layers=('3', '5', '7'),
                             shortcut_channels=(48, 120, 352),
                             fpn_type=BiFPN5, fpn_channels=192, **kwargs)

