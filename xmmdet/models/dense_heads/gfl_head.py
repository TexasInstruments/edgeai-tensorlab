import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.gfl_head import GFLHead

from pytorch_jacinto_ai import xnn
from ...ops import ConvModuleWrapper

@HEADS.register_module()
class GFLLiteHead(GFLHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModuleWrapper(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModuleWrapper(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = ConvModuleWrapper(
            self.feat_channels, self.cls_out_channels, 3, padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.gfl_reg = ConvModuleWrapper(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        # Focal Loss for Dense Object Detection, Tsung-Yi Lin et.al.
        # (RetinaNet paper): https://arxiv.org/pdf/1708.02002.pdf
        # initialization of class bias: sections 3.3, 4.1, 5.1
        # retinanet has only one shared cls head. get the last conv from it
        last_ms = xnn.layers.get_last_bias_modules(self.gfl_cls)
        bias_cls = bias_init_with_prob(0.01)
        if len(last_ms) > 0:
            bias_cls = bias_cls / len(last_ms)
            for m in last_ms:
                normal_init(m, std=0.01, bias=bias_cls)
            #
        #

