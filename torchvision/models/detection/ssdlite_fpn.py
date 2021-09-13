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

import torch
import warnings

from collections import OrderedDict
from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import _utils as det_utils
from .ssd import SSD, SSDScoringHead
from .anchor_utils import DefaultBoxGenerator
from .backbone_utils import _validate_trainable_layers
from .. import mobilenet
from ..mobilenetv3 import ConvBNActivation
from ..._internally_replaced_utils import load_state_dict_from_url
from .backbone_utils import BackboneWithFPN
from ...ops.feature_pyramid_network import LastLevelP6P7, FeaturePyramidNetwork
from ...ops.rf_blocks import BiFPN
from ...edgeailite import xnn

__all__ = ['ssdlite_mobilenet_v2_lite_fpn', 'ssdlite_mobilenet_v3_lite_large_fpn', 'ssdlite_mobilenet_v3_lite_small_fpn',
           'ssdlite_mobilenet_v2_lite_bifpn']


from .ssdlite import _normal_init, SSDLiteHead, _load_state_dict, model_urls


def ssdlite_fpn_model(pretrained: bool = False, progress: bool = True, num_classes: int = 91,
                      pretrained_backbone: bool = False, trainable_backbone_layers: Optional[int] = None,
                      norm_layer: Optional[Callable[..., nn.Module]] = None,
                      activation_layer: Callable[..., nn.Module] = nn.ReLU,
                      backbone_name = None, size = (320, 320), reduce_tail=False,
                      conv_cfg: dict = None, shortcut_layers = ('7', '14', '16'),
                      shortcut_channels = (80, 160, 960), fpn_type=FeaturePyramidNetwork,
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
    if "size" is None:
        warnings.warn("The size of the model is not provided; using default.")
        size = (320, 320)

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    # reduce_tail = not pretrained_backbone

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained_backbone, progress=progress,
                                                 norm_layer=norm_layer, **kwargs)

    if not pretrained_backbone:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)

    return_layers = {s:s for s in shortcut_layers}
    # in FeaturePyramidNetwork extra_blocks are applied at the output.
    # in BiFPN they are applied at the input
    p5_channels = 256 if fpn_type == FeaturePyramidNetwork else shortcut_channels[-1]
    extra_blocks = LastLevelP6P7(p5_channels,256,num_blocks=3,conv_cfg=conv_cfg)
    backbone = BackboneWithFPN(backbone.features, return_layers, in_channels_list=shortcut_channels,
                               out_channels=256, conv_cfg=conv_cfg, extra_blocks=extra_blocks,
                               fpn_type=fpn_type)

    # these factors are chosen to match those in mmdetection
    aspect_ratios = [[2]] + [[2,3]]*4 + [[2]]
    anchor_generator = DefaultBoxGenerator(aspect_ratios, min_ratio=0.1, max_ratio=0.9)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, -1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    if "image_mean" in kwargs:
        del defaults["image_mean"]
    #
    if "image_std" in kwargs:
        del defaults["image_std"]
    #
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, size, num_classes,
                head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer, activation_layer=activation_layer, conv_cfg=conv_cfg),
                **kwargs)

    if pretrained is True:
        weights_name = 'ssdlite320_mobilenet_v3_large_coco'
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


def ssdlite_mobilenet_v2_lite_fpn(*args, backbone_name="mobilenet_v2_lite", **kwargs):
    return ssdlite_fpn_model(*args, backbone_name=backbone_name, shortcut_layers=('6', '10', '18'),
                             shortcut_channels=(32, 64, 1280), conv_cfg=dict(group_size_dw=1), **kwargs)


def ssdlite_mobilenet_v3_lite_large_fpn(*args, backbone_name="mobilenet_v3_lite_large", **kwargs):
    return ssdlite_fpn_model(*args, backbone_name=backbone_name, shortcut_layers=('6', '12', '16'),
                             shortcut_channels=(40, 112, 960), conv_cfg=dict(group_size_dw=1), **kwargs)


def ssdlite_mobilenet_v3_lite_small_fpn(*args, backbone_name="mobilenet_v3_lite_small", **kwargs):
    return ssdlite_fpn_model(*args, backbone_name=backbone_name, shortcut_layers=('6', '12', '16'),
                             shortcut_channels=(40, 112, 960), conv_cfg=dict(group_size_dw=1), **kwargs)


def ssdlite_mobilenet_v2_lite_bifpn(*args, backbone_name="mobilenet_v2_lite", **kwargs):
    BiFPN2 = partial(BiFPN, num_blocks=2)
    return ssdlite_fpn_model(*args, backbone_name=backbone_name, shortcut_layers=('6', '10', '18'),
                             shortcut_channels=(32, 64, 1280), conv_cfg=dict(group_size_dw=1),
                             fpn_type=BiFPN2, with_iou_loss=False, **kwargs)

