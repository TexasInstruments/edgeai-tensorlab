import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.fcos_head import FCOSHead
from ...ops import ConvModuleWrapper

INF = 1e8


@HEADS.register_module()
class JaiFCOSHead(FCOSHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    Example:
        >>> self = JaiFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def _init_layers(self):
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
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModuleWrapper(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = ConvModuleWrapper(self.feat_channels, self.cls_out_channels, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
        self.fcos_reg = ConvModuleWrapper(self.feat_channels, 4, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)
        self.fcos_centerness = ConvModuleWrapper(self.feat_channels, 1, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=None, act_cfg=None)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            if hasattr(m,'conv') and hasattr(m.conv, 'weight'):
                normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            if hasattr(m,'conv') and hasattr(m.conv, 'weight'):
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        if hasattr(self.fcos_cls, 'weight'):
            normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        if hasattr(self.fcos_reg, 'weight'):
            normal_init(self.fcos_reg, std=0.01)
        if hasattr(self.fcos_centerness, 'weight'):
            normal_init(self.fcos_centerness, std=0.01)
