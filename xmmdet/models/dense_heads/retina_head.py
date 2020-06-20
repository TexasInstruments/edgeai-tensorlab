import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.retina_head import RetinaHead
from ...ops import ConvModuleWrapper


@HEADS.register_module()
class JaiRetinaHead(RetinaHead):
    """An anchor-based head used in
    `RetinaNet <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = JaiRetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
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
        self.retina_cls = ConvModuleWrapper(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.retina_reg = ConvModuleWrapper(
            self.feat_channels, self.num_anchors * 4, 3, padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

    def init_weights(self):
        for m in self.cls_convs:
            if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
                normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        if hasattr(self.retina_cls, 'weight'):
            normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        if hasattr(self.retina_reg, 'weight'):
            normal_init(self.retina_reg, std=0.01)



