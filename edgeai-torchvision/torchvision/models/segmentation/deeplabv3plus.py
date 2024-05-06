from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from ...edgeailite import xnn
from ._utils import _SimpleSegmentationModel
from .deeplabv3 import ASPP

__all__ = ["DeepLabV3Plus"]


class DeepLabV3Plus(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(DeepLabV3Plus, self).__init__(backbone, classifier, aux_classifier, dict_featues=True)


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, in_channels, num_classes, shortcut_channels, refine_decoder=True, conv_cfg=None, **kwargs):
        super(DeepLabV3PlusHead, self).__init__()
        self.refine_decoder = refine_decoder
        self.aspp = ASPP(in_channels, [6, 12, 18], conv_cfg=conv_cfg)
        self.shortcut = torch.nn.Sequential(
            nn.Conv2d(shortcut_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.cat = nn.Sequential(
            xnn.layers.CatBlock(),
            nn.Dropout(0.1)
        )
        if self.refine_decoder:
            self.refine = nn.Sequential(
                xnn.layers.ConvWrapper2d(304, 256, kernel_size=3, padding=1, bias=False, conv_cfg=conv_cfg),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                xnn.layers.ConvWrapper2d(256, 256, kernel_size=3, padding=1, bias=False, conv_cfg=conv_cfg),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        #
        self.project = nn.Sequential(
            nn.Conv2d(256 if self.refine_decoder else 304, num_classes, 1, bias=True)
        )

    def forward(self, features):
        assert len(features) >= 2, 'features must be of size at least 2'
        features = list(features.values()) if isinstance(features, (dict,OrderedDict)) else features
        xs, x = features[:2]
        x = self.aspp(x)
        x = F.interpolate(x, size=xs.shape[-2:], mode='bilinear', align_corners=False)
        xs = self.shortcut(xs)
        x = self.cat((x,xs))
        if self.refine_decoder:
            x = self.refine(x)
        #
        x = self.project(x)
        return x
